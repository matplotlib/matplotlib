"""
Place a legend on the axes at location loc.  Labels are a
sequence of strings and loc can be a string or an integer
specifying the legend location

The location codes are

  'best'         : 0,  (currently not supported, defaults to upper right)
  'upper right'  : 1,  (default)
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
import sys
from numerix import array, ones, Float


from matplotlib import verbose
from artist import Artist
from cbook import enumerate, is_string_like, iterable, silent_list
from font_manager import FontProperties
from lines import Line2D
from mlab import linspace
from patches import Patch, Rectangle, Shadow, bbox_artist, draw_bbox
from collections import LineCollection
from text import Text
from transforms import Bbox, Point, Value, get_bbox_transform, bbox_all,\
     unit_bbox, inverse_transform_bbox

class Legend(Artist):
    """
    Place a legend on the axes at location loc.  Labels are a
    sequence of strings and loc can be a string or an integer
    specifying the legend location

    The location codes are

      'best'         : 0,  (currently not supported, defaults to upper right)
      'upper right'  : 1,  (default)
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


    codes = {'best'         : 0,
             'upper right'  : 1,  # default
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




    def __init__(self, parent, handles, labels, loc,
                 isaxes=True,
                 numpoints = 4,      # the number of points in the legend line
                 prop = FontProperties(size='smaller'),
                 pad = 0.2,          # the fractional whitespace inside the legend border
                 markerscale = 0.6,    # the relative size of legend markers vs. original
                 # the following dimensions are in axes coords
                 labelsep = 0.005,     # the vertical space between the legend entries
                 handlelen = 0.05,     # the length of the legend lines
                 handletextsep = 0.02, # the space between the legend line and legend text
                 axespad = 0.02,       # the border between the axes and legend edge

                 shadow=False,
                 ):
        """
  parent                # the artist that contains the legend
  handles               # a list of artists (lines, patches) to add to the legend
  labels                # a list of strings to label the legend 
  loc                   # a location code
  isaxes=True           # whether this is an axes legend
  numpoints = 4         # the number of points in the legend line
  fontprop = FontProperties('smaller')  # the font property
  pad = 0.2             # the fractional whitespace inside the legend border
  markerscale = 0.6     # the relative size of legend markers vs. original
  shadow                # if True, draw a shadow behind legend 

The following dimensions are in axes coords
  labelsep = 0.005     # the vertical space between the legend entries
  handlelen = 0.05     # the length of the legend lines
  handletextsep = 0.02 # the space between the legend line and legend text
  axespad = 0.02       # the border between the axes and legend edge
        """
        Artist.__init__(self)
        if is_string_like(loc) and not self.codes.has_key(loc):
            verbose.report_error('Unrecognized location %s. Falling back on upper right; valid locations are\n%s\t' %(loc, '\n\t'.join(self.codes.keys())))
        if is_string_like(loc): loc = self.codes.get(loc, 1)
        
        self.numpoints = numpoints
        self.prop = prop
        self.fontsize = prop.get_size_in_points()
        self.pad = pad
        self.markerscale = markerscale
        self.labelsep = labelsep
        self.handlelen = handlelen
        self.handletextsep = handletextsep
        self.axespad = axespad
        self.shadow = shadow
        
        if isaxes:  # parent is an Axes
            self.set_figure(parent.figure)
        else:        # parent is a Figure
            self.set_figure(parent)

        self.parent = parent
        self.set_transform( get_bbox_transform( unit_bbox(), parent.bbox) )
        self._loc = loc   

        # make a trial box in the middle of the axes.  relocate it
        # based on it's bbox
        left, upper = 0.5, 0.5
        if self.numpoints == 1:
            self._xdata = array([left + self.handlelen*0.5])
        else:
            self._xdata = linspace(left, left + self.handlelen, self.numpoints)
        textleft = left+ self.handlelen+self.handletextsep
        self.texts = self._get_texts(labels, textleft, upper)
        self.handles = self._get_handles(handles, self.texts)
        
        left, top = self.texts[-1].get_position()
        HEIGHT = self._approx_text_height()
        bottom = top-HEIGHT
        left -= self.handlelen + self.handletextsep + self.pad
        self.legendPatch = Rectangle(
            xy=(left, bottom), width=0.5, height=HEIGHT*len(self.texts),
            facecolor='w', edgecolor='k',
            )
        self._set_artist_props(self.legendPatch)
        self._drawFrame = True

    def _set_artist_props(self, a):
        a.set_figure(self.figure)
        a.set_transform(self._transform)
        
    def _approx_text_height(self):
        return self.fontsize/72.0*self.figure.dpi.get()/self.parent.bbox.height()

            
    def draw(self, renderer):
        if not self.get_visible(): return 
        renderer.open_group('legend')
        self._update_positions(renderer)
        if self._drawFrame:
            if self.shadow:
                shadow = Shadow(self.legendPatch, -0.005, -0.005)
                shadow.draw(renderer)
            self.legendPatch.draw(renderer)
        

        for h in self.handles:            
            if h is not None:
		h.draw(renderer)
        	if 0: bbox_artist(h, renderer)

        for t in self.texts:
            if 0: bbox_artist(t, renderer)
            t.draw(renderer)
        renderer.close_group('legend')
        #draw_bbox(self.save, renderer, 'g')
        #draw_bbox(self.ibox, renderer, 'r', self._transform)

    def _get_handle_text_bbox(self, renderer):
        'Get a bbox for the text and lines in axes coords'

        bboxesText = [t.get_window_extent(renderer) for t in self.texts]
        bboxesHandles = [h.get_window_extent(renderer) for h in self.handles if h is not None]


        bboxesAll = bboxesText
        bboxesAll.extend(bboxesHandles)
        bbox = bbox_all(bboxesAll)
        self.save = bbox

        ibox =  inverse_transform_bbox(self._transform, bbox)
        self.ibox = ibox

        return ibox
        
    def _get_handles(self, handles, texts):
        HEIGHT = self._approx_text_height()

        ret = []   # the returned legend lines

        for handle, label in zip(handles, texts):
            x, y = label.get_position()
            x -= self.handlelen + self.handletextsep
            if isinstance(handle, Line2D):
                ydata = (y-HEIGHT/2)*ones(self._xdata.shape, Float)
                legline = Line2D(self._xdata, ydata)
                legline.update_from(handle)
                self._set_artist_props(legline) # after update                
                legline.set_clip_box(None)
                legline.set_markersize(self.markerscale*legline.get_markersize())
                legline.set_data_clipping(False)
                ret.append(legline)
            elif isinstance(handle, Patch):

                p = Rectangle(xy=(min(self._xdata), y-3/4*HEIGHT),
                              width = self.handlelen, height=HEIGHT/2,
                              )
                p.update_from(handle)
                self._set_artist_props(p)
                p.set_clip_box(None)
                ret.append(p)
            elif isinstance(handle, LineCollection):
                ydata = (y-HEIGHT/2)*ones(self._xdata.shape, Float)
                legline = Line2D(self._xdata, ydata)
                self._set_artist_props(legline)
                legline.set_clip_box(None)
                lw = handle.get_linewidths()[0]
                style = handle.get_linewidths()[0]
                color = handle.get_colors()[0]                
                legline.set_color(color)
                legline.set_linewidth(lw)
                ret.append(legline)
                
	    else:
		ret.append(None)
                                               
        return ret

    def draw_frame(self, b):
        'b is a boolean.  Set draw frame to b'
        self._drawFrame = b

    def get_frame(self):
        'return the Rectangle instance used to frame the legend'
        return self.legendPatch

    def get_lines(self):
        'return a list of lines.Line2D instances in the legend'
        return [h for h in self.handles if isinstance(h, Line2D)]  

    def get_patches(self):
        'return a list of patch instances in the legend'
        return silent_list('Patch', [h for h in self.handles if isinstance(h, Patch)])

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
                horizontalalignment='left',
                )
            self._set_artist_props(text)
            ret.append(text)
            pos -= HEIGHT
            
        return ret

            
    def get_window_extent(self):
        return self.legendPatch.get_window_extent()


    def _offset(self, ox, oy):
        'Move all the artists by ox,oy (axes coords)'
        for t in self.texts:
            x,y = t.get_position()
            t.set_position( (x+ox, y+oy) )

        for h in self.handles:
            if isinstance(h, Line2D):
                x,y = h.get_xdata(), h.get_ydata()
                h.set_data( x+ox, y+oy)
            elif isinstance(h, Rectangle):
                h.xy[0] = h.xy[0] + ox
                h.xy[1] = h.xy[1] + oy

        x, y = self.legendPatch.get_x(), self.legendPatch.get_y()
        self.legendPatch.set_x(x+ox)
        self.legendPatch.set_y(y+oy)

    def _update_positions(self, renderer):
        # called from renderer to allow more precise estimates of
        # widths and heights with get_window_extent

        def get_tbounds(text):  #get text bounds in axes coords
            bbox = text.get_window_extent(renderer)
            bboxa = inverse_transform_bbox(self._transform, bbox)
            return bboxa.get_bounds()
            
        hpos = []
        for t, tabove in zip(self.texts[1:], self.texts[:-1]):
            x,y = t.get_position()
            l,b,w,h = get_tbounds(tabove)
            hpos.append( (b,h) )
            t.set_position( (x, b-0.1*h) )

        # now do the same for last line
        l,b,w,h = get_tbounds(self.texts[-1])
        hpos.append( (b,h) )
        
        for handle, tup in zip(self.handles, hpos):
            y,h = tup
            if isinstance(handle, Line2D):
                ydata = y*ones(self._xdata.shape, Float)            
                handle.set_ydata(ydata+h/2)
            elif isinstance(handle, Rectangle):
                handle.set_y(y+1/4*h)
                handle.set_height(h/2)

        # Set the data for the legend patch
        bbox = self._get_handle_text_bbox(renderer).deepcopy()
        bbox.scale(1 + self.pad, 1 + self.pad)
        l,b,w,h = bbox.get_bounds()
        self.legendPatch.set_bounds(l,b,w,h)

        BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = range(11)
        ox, oy = 0, 0                           # center


        if iterable(self._loc) and len(self._loc)==2:
            xo = self.legendPatch.get_x()
            yo = self.legendPatch.get_y()
            x, y = self._loc
            ox = x-xo
            oy = y-yo
            self._offset(ox, oy)
        else:
            if self._loc in (UL, LL, CL):           # left
                ox = self.axespad - l
            if self._loc in (BEST, UR, LR, R, CR):  # right
                ox = 1 - (l + w + self.axespad)
            if self._loc in (BEST, UR, UL, UC):     # upper
                oy = 1 - (b + h + self.axespad)
            if self._loc in (LL, LR, LC):           # lower
                oy = self.axespad - b
            if self._loc in (LC, UC, C):            # center x
                ox = (0.5-w/2)-l
            if self._loc in (CL, CR, C):            # center y
                oy = (0.5-h/2)-b
            self._offset(ox, oy)
        
