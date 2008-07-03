"""
Place a legend on the axes at location loc.  Labels are a
sequence of strings and loc can be a string or an integer
specifying the legend location

The location codes are

  'best'         : 0, (only implemented for axis legends)
  'upper right'  : 1,
  'upper left'   : 2,
  'lower left'   : 3,
  'lower right'  : 4,
  'right'        : 5,
  'center left'  : 6,
  'center right' : 7,
  'lower center' : 8,
  'upper center' : 9,
  'center'       : 10,

Return value is a sequence of text, line instances that make
up the legend
"""
from __future__ import division
import warnings

import numpy as np

from matplotlib import rcParams
from artist import Artist
from cbook import is_string_like, iterable, silent_list, safezip
from font_manager import FontProperties
from lines import Line2D
from mlab import segments_intersect
from patches import Patch, Rectangle, Shadow, bbox_artist
from collections import LineCollection, RegularPolyCollection
from text import Text
from transforms import Affine2D, Bbox, BboxTransformTo

class Legend(Artist):
    """
    Place a legend on the axes at location loc.  Labels are a
    sequence of strings and loc can be a string or an integer
    specifying the legend location

    The location codes are

      'best'         : 0, (only implemented for axis legends)
      'upper right'  : 1,
      'upper left'   : 2,
      'lower left'   : 3,
      'lower right'  : 4,
      'right'        : 5,
      'center left'  : 6,
      'center right' : 7,
      'lower center' : 8,
      'upper center' : 9,
      'center'       : 10,

    Return value is a sequence of text, line instances that make
    up the legend
    """


    codes = {'best'         : 0, # only implemented for axis legends
             'upper right'  : 1,
             'upper left'   : 2,
             'lower left'   : 3,
             'lower right'  : 4,
             'right'        : 5,
             'center left'  : 6,
             'center right' : 7,
             'lower center' : 8,
             'upper center' : 9,
             'center'       : 10,
             }



    zorder = 5
    def __str__(self):
        return "Legend"

    def __init__(self, parent, handles, labels,
                 loc = None,
                 numpoints = None,     # the number of points in the legend line
                 prop = None,
                 pad = None,           # the fractional whitespace inside the legend border
                 markerscale = None,   # the relative size of legend markers vs. original
                 # the following dimensions are in axes coords
                 labelsep = None,      # the vertical space between the legend entries
                 handlelen = None,     # the length of the legend lines
                 handletextsep = None, # the space between the legend line and legend text
                 axespad = None,       # the border between the axes and legend edge

                 shadow = None
                 ):
        """
  parent                # the artist that contains the legend
  handles               # a list of artists (lines, patches) to add to the legend
  labels                # a list of strings to label the legend
  loc                   # a location code
  numpoints = 4         # the number of points in the legend line
  prop = FontProperties(size='smaller')  # the font property
  pad = 0.2             # the fractional whitespace inside the legend border
  markerscale = 0.6     # the relative size of legend markers vs. original
  shadow                # if True, draw a shadow behind legend

The following dimensions are in axes coords
  labelsep = 0.005     # the vertical space between the legend entries
  handlelen = 0.05     # the length of the legend lines
  handletextsep = 0.02 # the space between the legend line and legend text
  axespad = 0.02       # the border between the axes and legend edge
        """
        from axes import Axes     # local import only to avoid circularity
        from figure import Figure # local import only to avoid circularity

        Artist.__init__(self)

        proplist=[numpoints, pad, markerscale, labelsep, handlelen, handletextsep, axespad, shadow]
        propnames=['numpoints', 'pad', 'markerscale', 'labelsep', 'handlelen', 'handletextsep', 'axespad', 'shadow']
        for name, value in safezip(propnames,proplist):
            if value is None:
                value=rcParams["legend."+name]
            setattr(self,name,value)
        if self.numpoints <= 0:
            raise ValueError("numpoints must be >= 0; it was %d"% numpoints)
        if prop is None:
            self.prop=FontProperties(size=rcParams["legend.fontsize"])
        else:
            self.prop=prop
        self.fontsize = self.prop.get_size_in_points()

        if isinstance(parent,Axes):
            self.isaxes = True
            self.set_figure(parent.figure)
        elif isinstance(parent,Figure):
            self.isaxes = False
            self.set_figure(parent)
        else:
            raise TypeError("Legend needs either Axes or Figure as parent")
        self.parent = parent
        self._offsetTransform = Affine2D()
        self._parentTransform = BboxTransformTo(parent.bbox)
        Artist.set_transform(self, self._offsetTransform + self._parentTransform)

        if loc is None:
            loc = rcParams["legend.loc"]
            if not self.isaxes and loc in [0,'best']:
                loc = 'upper right'
        if is_string_like(loc):
            if not self.codes.has_key(loc):
                if self.isaxes:
                    warnings.warn('Unrecognized location "%s". Falling back on "best"; '
                                  'valid locations are\n\t%s\n'
                                  % (loc, '\n\t'.join(self.codes.keys())))
                    loc = 0
                else:
                    warnings.warn('Unrecognized location "%s". Falling back on "upper right"; '
                                  'valid locations are\n\t%s\n'
                                   % (loc, '\n\t'.join(self.codes.keys())))
                    loc = 1
            else:
                loc = self.codes[loc]
        if not self.isaxes and loc == 0:
            warnings.warn('Automatic legend placement (loc="best") not implemented for figure legend. '
                          'Falling back on "upper right".')
            loc = 1

        self._loc = loc

        self.legendPatch = Rectangle(
            xy=(0.0, 0.0), width=0.5, height=0.5,
            facecolor='w', edgecolor='k',
            )
        self._set_artist_props(self.legendPatch)

        # make a trial box in the middle of the axes.  relocate it
        # based on it's bbox
        left, top = 0.5, 0.5
        textleft = left+ self.handlelen+self.handletextsep
        self.texts = self._get_texts(labels, textleft, top)
        self.legendHandles = self._get_handles(handles, self.texts)

        self._drawFrame = True

    def _set_artist_props(self, a):
        a.set_figure(self.figure)
        a.set_transform(self.get_transform())

    def _approx_text_height(self):
        return self.fontsize/72.0*self.figure.dpi/self.parent.bbox.height


    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group('legend')
        self._update_positions(renderer)
        if self._drawFrame:
            if self.shadow:
                shadow = Shadow(self.legendPatch, -0.005, -0.005)
                shadow.draw(renderer)
            self.legendPatch.draw(renderer)


        if not len(self.legendHandles) and not len(self.texts): return
        for h in self.legendHandles:
            if h is not None:
                h.draw(renderer)
                if hasattr(h, '_legmarker'):
                    h._legmarker.draw(renderer)
                if 0: bbox_artist(h, renderer)

        for t in self.texts:
            if 0: bbox_artist(t, renderer)
            t.draw(renderer)
        renderer.close_group('legend')
        #draw_bbox(self.save, renderer, 'g')
        #draw_bbox(self.ibox, renderer, 'r', self.get_transform())

    def _get_handle_text_bbox(self, renderer):
        'Get a bbox for the text and lines in axes coords'

        bboxesText = [t.get_window_extent(renderer) for t in self.texts]
        bboxesHandles = [h.get_window_extent(renderer) for h in self.legendHandles if h is not None]

        bboxesAll = bboxesText
        bboxesAll.extend(bboxesHandles)
        bbox = Bbox.union(bboxesAll)

        self.save = bbox

        ibox = bbox.inverse_transformed(self.get_transform())
        self.ibox = ibox

        return ibox

    def _get_handles(self, handles, texts):
        handles = list(handles)
        texts = list(texts)
        HEIGHT = self._approx_text_height()
        left = 0.5

        ret = []   # the returned legend lines

        # we need to pad the text with empties for the numpoints=1
        # centered marker proxy

        for handle, label in safezip(handles, texts):
            if self.numpoints > 1:
                xdata = np.linspace(left, left + self.handlelen, self.numpoints)
                xdata_marker = xdata
            elif self.numpoints == 1:
                xdata = np.linspace(left, left + self.handlelen, 2)
                xdata_marker = [left + 0.5*self.handlelen]

            x, y = label.get_position()
            x -= self.handlelen + self.handletextsep
            if isinstance(handle, Line2D):
                ydata = (y-HEIGHT/2)*np.ones(xdata.shape, float)
                legline = Line2D(xdata, ydata)

                legline.update_from(handle)
                self._set_artist_props(legline) # after update
                legline.set_clip_box(None)
                legline.set_clip_path(None)
                ret.append(legline)
                legline.set_marker('None')

                legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
                legline_marker.update_from(handle)
                legline_marker.set_linestyle('None')
                self._set_artist_props(legline_marker)
                # we don't want to add this to the return list because
                # the texts and handles are assumed to be in one-to-one
                # correpondence.
                legline._legmarker = legline_marker

            elif isinstance(handle, Patch):
                p = Rectangle(xy=(min(xdata), y-3/4*HEIGHT),
                              width = self.handlelen, height=HEIGHT/2,
                              )
                p.update_from(handle)
                self._set_artist_props(p)
                p.set_clip_box(None)
                p.set_clip_path(None)
                ret.append(p)
            elif isinstance(handle, LineCollection):
                ydata = (y-HEIGHT/2)*np.ones(xdata.shape, float)
                legline = Line2D(xdata, ydata)
                self._set_artist_props(legline)
                legline.set_clip_box(None)
                legline.set_clip_path(None)
                lw = handle.get_linewidth()[0]
                dashes = handle.get_dashes()[0]
                color = handle.get_colors()[0]
                legline.set_color(color)
                legline.set_linewidth(lw)
                legline.set_dashes(dashes)
                ret.append(legline)

            elif isinstance(handle, RegularPolyCollection):
                if self.numpoints == 1:
                    xdata = np.array([left])
                p = Rectangle(xy=(min(xdata), y-3/4*HEIGHT),
                              width = self.handlelen, height=HEIGHT/2,
                              )
                p.set_facecolor(handle._facecolors[0])
                if handle._edgecolors != 'none' and len(handle._edgecolors):
                    p.set_edgecolor(handle._edgecolors[0])
                self._set_artist_props(p)
                p.set_clip_box(None)
                p.set_clip_path(None)
                ret.append(p)

            else:
                ret.append(None)

        return ret

    def _auto_legend_data(self):
        """ Returns list of vertices and extents covered by the plot.

        Returns a two long list.

        First element is a list of (x, y) vertices (in
        axes-coordinates) covered by all the lines and line
        collections, in the legend's handles.

        Second element is a list of bounding boxes for all the patches in
        the legend's handles.
        """

        assert self.isaxes # should always hold because function is only called internally

        ax = self.parent
        vertices = []
        bboxes = []
        lines = []

        inverse_transform = ax.transAxes.inverted()

        for handle in ax.lines:
            assert isinstance(handle, Line2D)
            path = handle.get_path()
            trans = handle.get_transform()
            tpath = trans.transform_path(path)
            apath = inverse_transform.transform_path(tpath)
            lines.append(apath)

        for handle in ax.patches:
            assert isinstance(handle, Patch)

            if isinstance(handle, Rectangle):
                transform = handle.get_data_transform() + inverse_transform
                bboxes.append(handle.get_bbox().transformed(transform))
            else:
                transform = handle.get_transform() + inverse_transform
                bboxes.append(handle.get_path().get_extents(transform))

        return [vertices, bboxes, lines]

    def draw_frame(self, b):
        'b is a boolean.  Set draw frame to b'
        self._drawFrame = b

    def get_children(self):
        children = []
        children.extend(self.legendHandles)
        children.extend(self.texts)
        return children

    def get_frame(self):
        'return the Rectangle instance used to frame the legend'
        return self.legendPatch

    def get_lines(self):
        'return a list of lines.Line2D instances in the legend'
        return [h for h in self.legendHandles if isinstance(h, Line2D)]

    def get_patches(self):
        'return a list of patch instances in the legend'
        return silent_list('Patch', [h for h in self.legendHandles if isinstance(h, Patch)])

    def get_texts(self):
        'return a list of text.Text instance in the legend'
        return silent_list('Text', self.texts)

    def _get_texts(self, labels, left, upper):

        # height in axes coords
        HEIGHT = self._approx_text_height()
        pos = upper
        x = left

        ret = []  # the returned list of text instances
        for l in labels:
            text = Text(
                x=x, y=pos,
                text=l,
                fontproperties=self.prop,
                verticalalignment='top',
                horizontalalignment='left'
                )
            self._set_artist_props(text)
            ret.append(text)
            pos -= HEIGHT

        return ret


    def get_window_extent(self):
        return self.legendPatch.get_window_extent()


    def _offset(self, ox, oy):
        'Move all the artists by ox,oy (axes coords)'
        self._offsetTransform.clear().translate(ox, oy)

    def _find_best_position(self, width, height, consider=None):
        """Determine the best location to place the legend.

        `consider` is a list of (x, y) pairs to consider as a potential
        lower-left corner of the legend. All are axes coords.
        """

        assert self.isaxes # should always hold because function is only called internally

        verts, bboxes, lines = self._auto_legend_data()

        consider = [self._loc_to_axes_coords(x, width, height) for x in range(1, len(self.codes))]

        tx, ty = self.legendPatch.get_x(), self.legendPatch.get_y()

        candidates = []
        for l, b in consider:
            legendBox = Bbox.from_bounds(l, b, width, height)
            badness = 0
            badness = legendBox.count_contains(verts)
            badness += legendBox.count_overlaps(bboxes)
            for line in lines:
                if line.intersects_bbox(legendBox):
                    badness += 1

            ox, oy = l-tx, b-ty
            if badness == 0:
                return ox, oy

            candidates.append((badness, (ox, oy)))

        # rather than use min() or list.sort(), do this so that we are assured
        # that in the case of two equal badnesses, the one first considered is
        # returned.
        minCandidate = candidates[0]
        for candidate in candidates:
            if candidate[0] < minCandidate[0]:
                minCandidate = candidate

        ox, oy = minCandidate[1]

        return ox, oy


    def _loc_to_axes_coords(self, loc, width, height):
        """Convert a location code to axes coordinates.

        - loc: a location code in range(1, 11).
          This corresponds to the possible values for self._loc, excluding "best".

        - width, height: the final size of the legend, axes units.
        """
        assert loc in range(1,11) # called only internally

        BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = range(11)

        if loc in (UL, LL, CL):                # left
            x = self.axespad
        elif loc in (UR, LR, CR, R):           # right
            x = 1.0 - (width + self.axespad)
        elif loc in (LC, UC, C):               # center x
            x = (0.5 - width/2.0)

        if loc in (UR, UL, UC):                # upper
            y = 1.0 - (height + self.axespad)
        elif loc in (LL, LR, LC):              # lower
            y = self.axespad
        elif loc in (CL, CR, C, R):            # center y
            y = (0.5 - height/2.0)

        return x,y


    def _update_positions(self, renderer):
        # called from renderer to allow more precise estimates of
        # widths and heights with get_window_extent

        if not len(self.legendHandles) and not len(self.texts): return
        def get_tbounds(text):  #get text bounds in axes coords
            bbox = text.get_window_extent(renderer)
            bboxa = bbox.inverse_transformed(self.get_transform())
            return bboxa.bounds

        hpos = []
        for t, tabove in safezip(self.texts[1:], self.texts[:-1]):
            x,y = t.get_position()
            l,b,w,h = get_tbounds(tabove)
            b -= self.labelsep
            h += 2*self.labelsep
            hpos.append( (b,h) )
            t.set_position( (x, b-0.1*h) )

        # now do the same for last line

        l,b,w,h = get_tbounds(self.texts[-1])
        b -= self.labelsep
        h += 2*self.labelsep
        hpos.append( (b,h) )

        for handle, tup in safezip(self.legendHandles, hpos):
            y,h = tup
            if isinstance(handle, Line2D):
                ydata = y*np.ones(handle.get_xdata().shape, float)
                handle.set_ydata(ydata+h/2.)
                handle._legmarker.set_ydata(ydata+h/2.)
            elif isinstance(handle, Rectangle):
                handle.set_y(y+1/4*h)
                handle.set_height(h/2)

        # Set the data for the legend patch
        bbox = self._get_handle_text_bbox(renderer)

        bbox = bbox.expanded(1 + self.pad, 1 + self.pad)
        l, b, w, h = bbox.bounds
        self.legendPatch.set_bounds(l, b, w, h)

        ox, oy = 0, 0                           # center

        if iterable(self._loc) and len(self._loc)==2:
            xo = self.legendPatch.get_x()
            yo = self.legendPatch.get_y()
            x, y = self._loc
            ox, oy = x-xo, y-yo
        elif self._loc == 0:  # "best"
            ox, oy = self._find_best_position(w, h)
        else:
            x, y = self._loc_to_axes_coords(self._loc, w, h)
            ox, oy = x-l, y-b

        self._offset(ox, oy)

#artist.kwdocd['Legend'] = kwdoc(Legend)
