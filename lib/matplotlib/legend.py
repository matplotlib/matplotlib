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
from matplotlib.artist import Artist
from matplotlib.cbook import is_string_like, iterable, silent_list, safezip
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Shadow, FancyBboxPatch
from matplotlib.collections import LineCollection, RegularPolyCollection
from matplotlib.transforms import Bbox

from matplotlib.offsetbox import HPacker, VPacker, PackerBase, TextArea, DrawingArea


class Legend(Artist):
    """
    Place a legend on the axes at location loc.  Labels are a
    sequence of strings and loc can be a string or an integer
    specifying the legend location

    The location codes are::

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

    loc can be a tuple of the noramilzed coordinate values with
    respect its parent.

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
                 markerscale = None,   # the relative size of legend markers vs. original
                 scatterpoints = 3,    # TODO: may be an rcParam
                 scatteryoffsets=None,
                 prop = None,          # properties for the legend texts

                 # the following dimensions are in axes coords
                 pad = None,           # deprecated; use borderpad
                 labelsep = None,      # deprecated; use labelspacing
                 handlelen = None,     # deprecated; use handlelength
                 handletextsep = None, # deprecated; use handletextpad
                 axespad = None,       # deprecated; use borderaxespad

                 # spacing & pad defined as a fractionof the font-size
                 borderpad = None,     # the whitespace inside the legend border
                 labelspacing=None, #the vertical space between the legend entries
                 handlelength=None, # the length of the legend handles
                 handletextpad=None, # the pad between the legend handle and text
                 borderaxespad=None, # the pad between the axes and legend border
                 columnspacing=None, # spacing between columns

                 ncol=1, # number of columns
                 mode=None, # mode for horizontal distribution of columns. None, "expand"

                 fancybox=None, # True use a fancy box, false use a rounded box, none use rc
                 shadow = None,
                 ):
        """
        - *parent* : the artist that contains the legend
        - *handles* : a list of artists (lines, patches) to add to the legend
        - *labels* : a list of strings to label the legend

        Optional keyword arguments:

        ================   ==================================================================
        Keyword            Description
        ================   ==================================================================
        loc                a location code or a tuple of coordinates
        numpoints          the number of points in the legend line
        prop               the font property
        markerscale        the relative size of legend markers vs. original
        fancybox           if True, draw a frame with a round fancybox.  If None, use rc
        shadow             if True, draw a shadow behind legend
        scatteryoffsets    a list of yoffsets for scatter symbols in legend
        borderpad          the fractional whitespace inside the legend border
        labelspacing       the vertical space between the legend entries
        handlelength       the length of the legend handles
        handletextpad      the pad between the legend handle and text
        borderaxespad      the pad between the axes and legend border
        columnspacing      the spacing between columns
        ================   ==================================================================

The dimensions of pad and spacing are given as a fraction of the
fontsize. Values from rcParams will be used if None.
        """
        from matplotlib.axes import Axes     # local import only to avoid circularity
        from matplotlib.figure import Figure # local import only to avoid circularity

        Artist.__init__(self)

        if prop is None:
            self.prop=FontProperties(size=rcParams["legend.fontsize"])
        else:
            self.prop=prop
        self.fontsize = self.prop.get_size_in_points()

        propnames=['numpoints', 'markerscale', 'shadow', "columnspacing",
                   "scatterpoints"]

        localdict = locals()

        for name in propnames:
            if localdict[name] is None:
                value = rcParams["legend."+name]
            else:
                value = localdict[name]
            setattr(self, name, value)

        # Take care the deprecated keywords
        deprecated_kwds = {"pad":"borderpad",
                           "labelsep":"labelspacing",
                           "handlelen":"handlelength",
                           "handletextsep":"handletextpad",
                           "axespad":"borderaxespad"}

        # convert values of deprecated keywords (ginve in axes coords)
        # to new vaules in a fraction of the font size

        # conversion factor
        bbox = parent.bbox
        axessize_fontsize = min(bbox.width, bbox.height)/self.fontsize

        for k, v in deprecated_kwds.items():
            # use deprecated value if not None and if their newer
            # counter part is None.
            if localdict[k] is not None and localdict[v] is None:
                warnings.warn("Use '%s' instead of '%s'." % (v, k),
                              DeprecationWarning)
                setattr(self, v, localdict[k]*axessize_fontsize)
                continue

            # Otherwise, use new keywords
            if localdict[v] is None:
                setattr(self, v, rcParams["legend."+v])
            else:
                setattr(self, v, localdict[v])

        del localdict

        self._ncol = ncol

        if self.numpoints <= 0:
            raise ValueError("numpoints must be >= 0; it was %d"% numpoints)

        # introduce y-offset for handles of the scatter plot
        if scatteryoffsets is None:
            self._scatteryoffsets = np.array([3./8., 4./8., 2.5/8.])
        else:
            self._scatteryoffsets = np.asarray(scatteryoffsets)
        reps =  int(self.numpoints / len(self._scatteryoffsets)) + 1
        self._scatteryoffsets = np.tile(self._scatteryoffsets, reps)[:self.scatterpoints]

        # _legend_box is an OffsetBox instance that contains all
        # legend items and will be initialized from _init_legend_box()
        # method.
        self._legend_box = None

        if isinstance(parent,Axes):
            self.isaxes = True
            self.set_figure(parent.figure)
        elif isinstance(parent,Figure):
            self.isaxes = False
            self.set_figure(parent)
        else:
            raise TypeError("Legend needs either Axes or Figure as parent")
        self.parent = parent

        if loc is None:
            loc = rcParams["legend.loc"]
            if not self.isaxes and loc in [0,'best']:
                loc = 'upper right'
        if is_string_like(loc):
            if loc not in self.codes:
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
        self._mode = mode

        # We use FancyBboxPatch to draw a legend frame. The location
        # and size of the box will be updated during the drawing time.
        self.legendPatch = FancyBboxPatch(
            xy=(0.0, 0.0), width=1., height=1.,
            facecolor='w', edgecolor='k',
            mutation_scale=self.fontsize,
            snap=True
            )

        # The width and height of the legendPatch will be set (in the
        # draw()) to the length that includes the padding. Thus we set
        # pad=0 here.
        if fancybox is None:
            fancybox = rcParams["legend.fancybox"]

        if fancybox == True:
            self.legendPatch.set_boxstyle("round",pad=0,
                                          rounding_size=0.2)
        else:
            self.legendPatch.set_boxstyle("square",pad=0)

        self._set_artist_props(self.legendPatch)

        self._drawFrame = True
        
        # init with null renderer
        self._init_legend_box(handles, labels)

        self._last_fontsize_points = self.fontsize


    def _set_artist_props(self, a):
        """
        set the boilerplate props for artists added to axes
        """
        a.set_figure(self.figure)

        for c in self.get_children():
            c.set_figure(self.figure)

        a.set_transform(self.get_transform())

    def _findoffset_best(self, width, height, xdescent, ydescent, renderer):
        "Heper function to locate the legend at its best position"
        ox, oy = self._find_best_position(width, height, renderer)
        return ox+xdescent, oy+ydescent

    def _findoffset_loc(self, width, height, xdescent, ydescent, renderer):
        "Heper function to locate the legend using the location code"

        if iterable(self._loc) and len(self._loc)==2:
            # when loc is a tuple of axes(or figure) coordinates.
            fx, fy = self._loc
            bbox = self.parent.bbox
            x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy
        else:
            bbox = Bbox.from_bounds(0, 0, width, height)
            x, y = self._get_anchored_bbox(self._loc, bbox, self.parent.bbox, renderer)

        return x+xdescent, y+ydescent

    def draw(self, renderer):
        "Draw everything that belongs to the legend"
        if not self.get_visible(): return

        self._update_legend_box(renderer)

        renderer.open_group('legend')

        # find_offset function will be provided to _legend_box and
        # _legend_box will draw itself at the location of the return
        # value of the find_offset.
        if self._loc == 0:
            _findoffset = self._findoffset_best
        else:
            _findoffset = self._findoffset_loc

        def findoffset(width, height, xdescent, ydescent):
            return _findoffset(width, height, xdescent, ydescent, renderer)
        
        self._legend_box.set_offset(findoffset)
        
        fontsize = renderer.points_to_pixels(self.fontsize)

        # if mode == fill, set the width of the legend_box to the
        # width of the paret (minus pads)
        if self._mode in ["expand"]:
            pad = 2*(self.borderaxespad+self.borderpad)*fontsize
            self._legend_box.set_width(self.parent.bbox.width-pad)

        if self._drawFrame:
            # update the location and size of the legend
            bbox = self._legend_box.get_window_extent(renderer)
            self.legendPatch.set_bounds(bbox.x0, bbox.y0,
                                        bbox.width, bbox.height)

            self.legendPatch.set_mutation_scale(fontsize)

            if self.shadow:
                shadow = Shadow(self.legendPatch, 2, -2)
                shadow.draw(renderer)

            self.legendPatch.draw(renderer)

        self._legend_box.draw(renderer)

        renderer.close_group('legend')


    def _approx_text_height(self, renderer=None):
        """
        Return the approximate height of the text. This is used to place
        the legend handle.
        """
        if renderer is None:
            return self.fontsize
        else:
            return renderer.points_to_pixels(self.fontsize)


    def _init_legend_box(self, handles, labels):
        """
        Initiallize the legend_box. The legend_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """

        fontsize = self.fontsize

        # legend_box is a HPacker, horizontally packed with
        # columns. Each column is a VPacker, vertically packed with
        # legend items. Each legend item is HPacker packed with
        # legend handleBox and labelBox. handleBox is an instance of
        # offsetbox.DrawingArea which contains legend handle. labelBox
        # is an instance of offsetbox.TextArea which contains legend
        # text.


        text_list = []  # the list of text instances
        handle_list = []  # the list of text instances

        label_prop = dict(verticalalignment='baseline',
                          horizontalalignment='left',
                          fontproperties=self.prop,
                          )

        labelboxes = []

        for l in labels:
            textbox = TextArea(l, textprops=label_prop,
                               multilinebaseline=True, minimumdescent=True)
            text_list.append(textbox._text)
            labelboxes.append(textbox)

        handleboxes = []


        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.
        height = self._approx_text_height() * 0.7
        descent = 0.

        # each handle needs to be drawn inside a box of (x, y, w, h) =
        # (0, -descent, width, height).  And their corrdinates should
        # be given in the display coordinates.

        # NOTE : the coordinates will be updated again in
        # _update_legend_box() method.

        # The transformation of each handle will be automatically set
        # to self.get_trasnform(). If the artist does not uses its
        # default trasnform (eg, Collections), you need to
        # manually set their transform to the self.get_transform().

        for handle in handles:
            if isinstance(handle, RegularPolyCollection):
                npoints = self.scatterpoints
            else:
                npoints = self.numpoints
            if npoints > 1:
                # we put some pad here to compensate the size of the
                # marker
                xdata = np.linspace(0.3*fontsize,
                                    (self.handlelength-0.3)*fontsize,
                                    npoints)
                xdata_marker = xdata
            elif npoints == 1:
                xdata = np.linspace(0, self.handlelength*fontsize, 2)
                xdata_marker = [0.5*self.handlelength*fontsize]

            if isinstance(handle, Line2D):
                ydata = ((height-descent)/2.)*np.ones(xdata.shape, float)
                legline = Line2D(xdata, ydata)

                legline.update_from(handle)
                self._set_artist_props(legline) # after update
                legline.set_clip_box(None)
                legline.set_clip_path(None)
                legline.set_drawstyle('default')
                legline.set_marker('None')

                handle_list.append(legline)

                legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
                legline_marker.update_from(handle)
                self._set_artist_props(legline_marker)
                legline_marker.set_clip_box(None)
                legline_marker.set_clip_path(None)
                legline_marker.set_linestyle('None')
                # we don't want to add this to the return list because
                # the texts and handles are assumed to be in one-to-one
                # correpondence.
                legline._legmarker = legline_marker

            elif isinstance(handle, Patch):
                p = Rectangle(xy=(0., 0.),
                              width = self.handlelength*fontsize,
                              height=(height-descent),
                              )
                p.update_from(handle)
                self._set_artist_props(p)
                p.set_clip_box(None)
                p.set_clip_path(None)
                handle_list.append(p)
            elif isinstance(handle, LineCollection):
                ydata = ((height-descent)/2.)*np.ones(xdata.shape, float)
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
                handle_list.append(legline)

            elif isinstance(handle, RegularPolyCollection):

                #ydata = self._scatteryoffsets
                ydata = height*self._scatteryoffsets

                size_max, size_min = max(handle.get_sizes()),\
                                     min(handle.get_sizes())
                # we may need to scale these sizes by "markerscale"
                # attribute. But other handle types does not seem
                # to care about this attribute and it is currently ignored.
                if self.scatterpoints < 4:
                    sizes = [.5*(size_max+size_min), size_max,
                             size_min]
                else:
                    sizes = (size_max-size_min)*np.linspace(0,1,self.scatterpoints)+size_min

                p = type(handle)(handle.get_numsides(),
                                 rotation=handle.get_rotation(),
                                 sizes=sizes,
                                 offsets=zip(xdata_marker,ydata),
                                 transOffset=self.get_transform(),
                                 )

                p.update_from(handle)
                p.set_figure(self.figure)
                p.set_clip_box(None)
                p.set_clip_path(None)
                handle_list.append(p)

            else:
                handle_list.append(None)

            handlebox = DrawingArea(width=self.handlelength*fontsize,
                                    height=height,
                                    xdescent=0., ydescent=descent)

            handle = handle_list[-1]
            handlebox.add_artist(handle)
            if hasattr(handle, "_legmarker"):
                handlebox.add_artist(handle._legmarker)
            handleboxes.append(handlebox)


        # We calculate number of lows in each column. The first
        # (num_largecol) columns will have (nrows+1) rows, and remaing
        # (num_smallcol) columns will have (nrows) rows.
        nrows, num_largecol = divmod(len(handleboxes), self._ncol)
        num_smallcol = self._ncol-num_largecol

        # starting index of each column and number of rows in it.
        largecol = safezip(range(0, num_largecol*(nrows+1), (nrows+1)),
                           [nrows+1] * num_largecol)
        smallcol = safezip(range(num_largecol*(nrows+1), len(handleboxes), nrows),
                           [nrows] * num_smallcol)

        handle_label = safezip(handleboxes, labelboxes)
        columnbox = []
        for i0, di in largecol+smallcol:
            # pack handleBox and labelBox into itemBox
            itemBoxes = [HPacker(pad=0,
                                 sep=self.handletextpad*fontsize,
                                 children=[h, t], align="baseline")
                         for h, t in handle_label[i0:i0+di]]
            # minimumdescent=False for the text of the last row of the column
            itemBoxes[-1].get_children()[1].set_minimumdescent(False)

            # pack columnBox
            columnbox.append(VPacker(pad=0,
                                        sep=self.labelspacing*fontsize,
                                        align="baseline",
                                        children=itemBoxes))

        if self._mode == "expand":
            mode = "expand"
        else:
            mode = "fixed"

        sep = self.columnspacing*fontsize

        self._legend_box = HPacker(pad=self.borderpad*fontsize,
                                      sep=sep, align="baseline",
                                      mode=mode,
                                      children=columnbox)

        self._legend_box.set_figure(self.figure)

        self.texts = text_list
        self.legendHandles = handle_list




    def _update_legend_box(self, renderer):
        """
        Update the dimension of the legend_box. This is required
        becuase the paddings, the hadle size etc. depends on the dpi
        of the renderer.
        """

        # fontsize in points.
        fontsize = renderer.points_to_pixels(self.fontsize)

        if self._last_fontsize_points == fontsize:
            # no update is needed
            return

        # each handle needs to be drawn inside a box of
        # (x, y, w, h) = (0, -descent, width, height).
        # And their corrdinates should be given in the display coordinates.

        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.
        height = self._approx_text_height(renderer) * 0.7
        descent = 0.

        for handle in self.legendHandles:
            if isinstance(handle, RegularPolyCollection):
                npoints = self.scatterpoints
            else:
                npoints = self.numpoints
            if npoints > 1:
                # we put some pad here to compensate the size of the
                # marker
                xdata = np.linspace(0.3*fontsize,
                                    (self.handlelength-0.3)*fontsize,
                                    npoints)
                xdata_marker = xdata
            elif npoints == 1:
                xdata = np.linspace(0, self.handlelength*fontsize, 2)
                xdata_marker = [0.5*self.handlelength*fontsize]

            if isinstance(handle, Line2D):
                legline = handle
                ydata = ((height-descent)/2.)*np.ones(xdata.shape, float)
                legline.set_data(xdata, ydata)

                legline_marker = legline._legmarker
                legline_marker.set_data(xdata_marker, ydata[:len(xdata_marker)])

            elif isinstance(handle, Patch):
                p = handle
                p.set_bounds(0., 0.,
                             self.handlelength*fontsize,
                             (height-descent),
                             )

            elif isinstance(handle, RegularPolyCollection):

                p = handle
                ydata = height*self._scatteryoffsets
                p.set_offsets(zip(xdata_marker,ydata))


        # correction factor
        cor = fontsize / self._last_fontsize_points

        # helper function to iterate over all children
        def all_children(parent):
            yield parent
            for c in parent.get_children():
                for cc in all_children(c): yield cc


        #now update paddings
        for box in all_children(self._legend_box):
            if isinstance(box, PackerBase):
                box.pad = box.pad * cor
                box.sep = box.sep * cor

            elif isinstance(box, DrawingArea):
                box.width = self.handlelength*fontsize
                box.height = height
                box.xdescent = 0.
                box.ydescent=descent

        self._last_fontsize_points = fontsize


    def _auto_legend_data(self):
        """
        Returns list of vertices and extents covered by the plot.

        Returns a two long list.

        First element is a list of (x, y) vertices (in
        display-coordinates) covered by all the lines and line
        collections, in the legend's handles.

        Second element is a list of bounding boxes for all the patches in
        the legend's handles.
        """

        assert self.isaxes # should always hold because function is only called internally

        ax = self.parent
        vertices = []
        bboxes = []
        lines = []

        for handle in ax.lines:
            assert isinstance(handle, Line2D)
            path = handle.get_path()
            trans = handle.get_transform()
            tpath = trans.transform_path(path)
            lines.append(tpath)

        for handle in ax.patches:
            assert isinstance(handle, Patch)

            if isinstance(handle, Rectangle):
                transform = handle.get_data_transform()
                bboxes.append(handle.get_bbox().transformed(transform))
            else:
                transform = handle.get_transform()
                bboxes.append(handle.get_path().get_extents(transform))

        return [vertices, bboxes, lines]

    def draw_frame(self, b):
        'b is a boolean.  Set draw frame to b'
        self._drawFrame = b

    def get_children(self):
        'return a list of child artists'
        children = []
        if self._legend_box:
            children.append(self._legend_box)
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

    def get_window_extent(self):
        'return a extent of the the legend'
        return self.legendPatch.get_window_extent()


    def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
        """
        Place the *bbox* inside the *parentbbox* according to a given
        location code. Return the (x,y) coordinate of the bbox.

        - loc: a location code in range(1, 11).
          This corresponds to the possible values for self._loc, excluding "best".

        - bbox: bbox to be placed, display coodinate units.
        - parentbbox: a parent box which will contain the bbox. In
            display coordinates.
        """
        assert loc in range(1,11) # called only internally

        BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = range(11)

        anchor_coefs={UR:"NE",
                      UL:"NW",
                      LL:"SW",
                      LR:"SE",
                      R:"E",
                      CL:"W",
                      CR:"E",
                      LC:"S",
                      UC:"N",
                      C:"C"}

        c = anchor_coefs[loc]
        
        fontsize = renderer.points_to_pixels(self.fontsize)
        container = parentbbox.padded(-(self.borderaxespad) * fontsize)
        anchored_box = bbox.anchored(c, container=container)
        return anchored_box.x0, anchored_box.y0


    def _find_best_position(self, width, height, renderer, consider=None):
        """
        Determine the best location to place the legend.

        `consider` is a list of (x, y) pairs to consider as a potential
        lower-left corner of the legend. All are display coords.
        """

        assert self.isaxes # should always hold because function is only called internally

        verts, bboxes, lines = self._auto_legend_data()

        bbox = Bbox.from_bounds(0, 0, width, height)
        consider = [self._get_anchored_bbox(x, bbox, self.parent.bbox, renderer) for x in range(1, len(self.codes))]

        #tx, ty = self.legendPatch.get_x(), self.legendPatch.get_y()

        candidates = []
        for l, b in consider:
            legendBox = Bbox.from_bounds(l, b, width, height)
            badness = 0
            badness = legendBox.count_contains(verts)
            badness += legendBox.count_overlaps(bboxes)
            for line in lines:
                if line.intersects_bbox(legendBox):
                    badness += 1

            ox, oy = l, b
            if badness == 0:
                return ox, oy

            candidates.append((badness, (l, b)))

        # rather than use min() or list.sort(), do this so that we are assured
        # that in the case of two equal badnesses, the one first considered is
        # returned.
        # NOTE: list.sort() is stable.But leave as it is for now. -JJL
        minCandidate = candidates[0]
        for candidate in candidates:
            if candidate[0] < minCandidate[0]:
                minCandidate = candidate

        ox, oy = minCandidate[1]

        return ox, oy

