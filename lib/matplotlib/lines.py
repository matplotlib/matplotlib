"""
This module contains all the 2D line class which can draw with a
variety of line styles, markers and colors
"""

# TODO: expose cap and join style attrs
from __future__ import division

import sys, math, warnings

import agg
from numerix import Float, alltrue, arange, array, logical_and,\
     nonzero, searchsorted, take, asarray, ones, where, less, ravel, \
     greater, logical_and, cos, sin, pi
from matplotlib import verbose
from artist import Artist
from cbook import iterable, is_string_like
from colors import colorConverter
from patches import bbox_artist

from transforms import lbwh_to_bbox, LOG10
from matplotlib import rcParams

TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN = range(4)
lineStyles  = {'-':1, '--':1, '-.':1, ':':1, 'steps':1, 'None':1}
lineMarkers =    {'.':1, ',':1, 'o':1, '^':1, 'v':1, '<':1, '>':1, 's':1,
                  '+':1, 'x':1, 'd':1, 'D':1, '|':1, '_':1, 'h':1, 'H':1,
                  'p':1, '1':1, '2':1, '3':1, '4':1,
                  TICKLEFT:1,
                  TICKRIGHT:1,
                  TICKUP:1,
                  TICKDOWN:1,
                  'None':1
                  }
    
class Line2D(Artist):
    _lineStyles =  {
        '-'    : '_draw_solid',
        '--'   : '_draw_dashed', 
        '-.'   : '_draw_dash_dot', 
        ':'    : '_draw_dotted',
        'steps': '_draw_steps',
        'None' : '_draw_nothing'}
    
    _markers =  {
        '.'  : '_draw_point',
        ','  : '_draw_pixel', 
        'o'  : '_draw_circle', 
        'v'  : '_draw_triangle_down', 
        '^'  : '_draw_triangle_up', 
        '<'  : '_draw_triangle_left', 
        '>'  : '_draw_triangle_right', 
        '1'  : '_draw_tri_down', 
        '2'  : '_draw_tri_up', 
        '3'  : '_draw_tri_left', 
        '4'  : '_draw_tri_right', 
        's'  : '_draw_square',
        'p'  : '_draw_pentagon',
        'h'  : '_draw_hexagon1',
        'H'  : '_draw_hexagon2',
        '+'  : '_draw_plus',
        'x'  : '_draw_x',
        'D'  : '_draw_diamond',
        'd'  : '_draw_thin_diamond',
        '|'  : '_draw_vline',
        '_'  : '_draw_hline',
        TICKLEFT    : '_draw_tickleft', 
        TICKRIGHT   : '_draw_tickright',
        TICKUP      : '_draw_tickup',
        TICKDOWN    : '_draw_tickdown',
        'None' : '_draw_nothing'
        }

    zorder = 2
        
    def __init__(self, xdata, ydata,
                 linewidth       = None, # default to rc
                 linestyle       = None, # default to rc
                 color           = None, # default to rc
                 marker          = None, # default to rc
                 markersize      = None, # default to rc
                 markeredgewidth = None, # default to rc
                 markeredgecolor = None, # default to rc
                 markerfacecolor = None, # default to rc
                 antialiased     = None, # default to rc
                 ):
        """
        xdata is a sequence of x data
        ydata is a sequence of y data
        linewidth is the width of the line in points
        linestyle is one of lineStyles
        marker is one of lineMarkers or None
        markersize is the size of the marker in points
        markeredgecolor  and markerfacecolor can be any color arg 
        markeredgewidth is the size of the markter edge in points

        """
        Artist.__init__(self)

        #convert sequences to numeric arrays
        if not iterable(xdata):
            raise RuntimeError('xdata must be a sequence')
        if not iterable(ydata):
            raise RuntimeError('ydata must be a sequence')

        if linewidth is None   : linewidth=rcParams['lines.linewidth']
        if linestyle is None   : linestyle=rcParams['lines.linestyle']
        if marker is None      : marker=rcParams['lines.marker']
        if color is None       : color=rcParams['lines.color']
        if markeredgecolor is None :
            markeredgecolor=rcParams['lines.markeredgecolor']
        if markerfacecolor is None :
            markerfacecolor=rcParams['lines.markerfacecolor']
        if markeredgewidth is None :
            markeredgewidth=rcParams['lines.markeredgewidth']
        
        if markersize is None  : markersize=rcParams['lines.markersize']
        if antialiased is None : antialiased=rcParams['lines.antialiased']


        self._linestyle = linestyle
        self._linewidth = linewidth
        self._color = color
        self._antialiased = antialiased
        self._markersize = markersize
        self._dashSeq = None


        self._markerfacecolor = markerfacecolor
        self._markeredgecolor = markeredgecolor
        self._markeredgewidth = markeredgewidth

        self.verticalOffset = None        
        self._useDataClipping = rcParams['lines.data_clipping']
        self.set_data(xdata, ydata)

        if not self._lineStyles.has_key(linestyle):
            raise ValueError('Unrecognized line style %s, %s' %( linestyle, type(linestyle)))
        if not self._markers.has_key(marker):
            raise ValueError('Unrecognized marker style %s, %s'%( marker, type(marker)))

        self.set_marker(marker)

        self._logcache = None
            
    def get_window_extent(self, renderer):
        self._newstyle = hasattr(renderer, 'draw_markers')
        if self._newstyle:
            x = self._x
            y = self._y
        else:
            x, y = self._get_numeric_clipped_data_in_range()


        x, y = self._transform.numerix_x_y(x, y)
        #x, y = self._transform.seq_x_y(x, y)

        left = min(x)
        bottom = min(y)
        width = max(x) - left
        height = max(y) - bottom

        # correct for marker size, if any
        if self._marker is not None:
            ms = self._markersize/72.0*self.figure.dpi.get()
            left -= ms/2
            bottom -= ms/2
            width += ms
            height += ms
        return lbwh_to_bbox( left, bottom, width, height)


    def set_data(self, *args):
        """
Set the x and y data

ACCEPTS: (array xdata, array ydata)
"""

        if len(args)==1:
            x, y = args[0]
        else:
            x, y = args


        try: del self._xc, self._yc
        except AttributeError: pass

        self._x = asarray(x, Float)
        self._y = asarray(y, Float)

        if len(self._x.shape)>1:
            self._x = ravel(self._x)
        if len(self._y.shape)>1:
            self._y = ravel(self._y)


        if len(self._y)==1 and len(self._x)>1:
            self._y = self._y*ones(self._x.shape, Float)

        if len(self._x) != len(self._y):
            raise RuntimeError('xdata and ydata must be the same length')

        if self._useDataClipping: self._xsorted = self._is_sorted(self._x)

        self._logcache = None
        
    def set_data_clipping(self, b):
        """
b is a boolean that sets whether data clipping is on

ACCEPTS: [True | False]
        """
        self._useDataClipping = b

                
    def _is_sorted(self, x):
        "return true if x is sorted"
        if len(x)<2: return 1
        return alltrue(x[1:]-x[0:-1]>=0)
    
    def _get_numeric_clipped_data_in_range(self):
        # if the x or y clip is set, only plot the points in the
        # clipping region.  If log scale is set, only pos data will be
        # returned

        try: self._xc, self._yc
        except AttributeError: x, y = self._x, self._y
        else: x, y = self._xc, self._yc

        try: logx = self._transform.get_funcx().get_type()==LOG10
        except RuntimeError: logx = False  # non-separable
        
        try: logy = self._transform.get_funcy().get_type()==LOG10
        except RuntimeError: logy = False  # non-separable
        
        if not logx and not logy:
            return x, y

        if self._logcache is not None:
            waslogx, waslogy, xcache, ycache = self._logcache
            if logx==waslogx and waslogy==logy:            
                return xcache, ycache
            
        Nx = len(x)
        Ny = len(y)

        if logx: indx = greater(x, 0)
        else:    indx = ones(len(x))

        if logy: indy = greater(y, 0)
        else:    indy = ones(len(y))

        ind = nonzero(logical_and(indx, indy))
        x = take(x, ind)
        y = take(y, ind)

        self._logcache = logx, logy, x, y
        return x, y

    def draw(self, renderer):
        #renderer.open_group('line2d')

        self._newstyle = hasattr(renderer, 'draw_markers')
        gc = renderer.new_gc()
        gc.set_foreground(self._color)
        gc.set_antialiased(self._antialiased)
        gc.set_linewidth(self._linewidth)
        gc.set_alpha(self._alpha)
        if self.get_clip_on():
            gc.set_clip_rectangle(self.clipbox.get_bounds())

        
        if self._newstyle:
            # transform in backend
            xt = self._x
            yt = self._y
        else:    
            x, y = self._get_numeric_clipped_data_in_range()
            if len(x)==0: return 
            xt, yt = self._transform.numerix_x_y(x, y)


        funcname = self._lineStyles.get(self._linestyle, '_draw_nothing')
        lineFunc = getattr(self, funcname)
        lineFunc(renderer, gc, xt, yt)


        if self._marker is not None:
            gc = renderer.new_gc()
            gc.set_foreground(self._markeredgecolor)
            gc.set_linewidth(self._markeredgewidth)
            if self.get_clip_on():
                gc.set_clip_rectangle(self.clipbox.get_bounds())
            funcname = self._markers.get(self._marker, '_draw_nothing')
            markerFunc = getattr(self, funcname)
            markerFunc(renderer, gc, xt, yt)

        #if 1: bbox_artist(self, renderer)
        #renderer.close_group('line2d')
        
    def get_antialiased(self): return self._antialiased
    def get_color(self): return self._color
    def get_linestyle(self): return self._linestyle

    def get_linewidth(self): return self._linewidth
    def get_marker(self): return self._marker
    def get_markeredgecolor(self): return self._markeredgecolor
    def get_markeredgewidth(self): return self._markeredgewidth
    def get_markerfacecolor(self): return self._markerfacecolor
    def get_markersize(self): return self._markersize
    def get_xdata(self): return self._x
    def get_ydata(self):  return self._y

        
    def _set_clip(self):

        
        if not self._useDataClipping: return
        #self._logcache = None
        
        try: self._xmin, self._xmax
        except AttributeError: indx = arange(len(self._x))
        else:
            if not hasattr(self, '_xsorted'):
                self._xsorted = self._is_sorted(self._x)
            if len(self._x)==1:
                indx = [0]
            elif self._xsorted:
                # for really long signals, if we know they are sorted
                # on x we can save a lot of time using search sorted
                # since the alternative approach requires 3 O(len(x) ) ops
                indMin, indMax = searchsorted(
                    self._x, array([self._xmin, self._xmax]))
                indMin = max(0, indMin-1)
                indMax = min(indMax+1, len(self._x))
                skip = 0
                if self._lod:
                    # if level of detail is on, decimate the data
                    # based on pixel width
                    raise NotImplementedError('LOD deprecated')
                    l, b, w, h = self.get_window_extent().get_bounds()
                    skip = int((indMax-indMin)/w)                    
                if skip>0:  indx = arange(indMin, indMax, skip)
                else: indx = arange(indMin, indMax)
            else:
                indx = nonzero(
                    logical_and( self._x>=self._xmin,
                                 self._x<=self._xmax ))

        self._xc = take(self._x, indx)
        self._yc = take(self._y, indx)

        # y data clipping for connected lines can introduce horizontal
        # line artifacts near the clip region.  If you really need y
        # clipping for efficiency, consider using plot(y,x) instead.
        if ( self._yc.shape==self._xc.shape and 
             self._linestyle is None):
            try: self._ymin, self._ymax
            except AttributeError: indy = arange(len(self._yc))
            else: indy = nonzero(
                logical_and(self._yc>=self._ymin,
                                  self._yc<=self._ymax ))
        else:
            indy = arange(len(self._yc))
            
        self._xc = take(self._xc, indy)
        self._yc = take(self._yc, indy)

    def set_antialiased(self, b):
        """
True if line should be drawin with antialiased rendering

ACCEPTS: [True | False]
        """            
        self._antialiased = b

    def set_color(self, color):
        """
Set the color of the line

ACCEPTS: any matplotlib color - see help(colors)
        """
        self._color = color

    def set_linewidth(self, w):
        """
Set the line width in points

ACCEPTS: float value in points
        """
        self._linewidth = w

    def set_linestyle(self, s):
        """
Set the linestyle of the line

ACCEPTS: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]
        """
        self._linestyle = s


    def set_marker(self, marker):
        """
Set the line marker

ACCEPTS: [ '+' | ',' | '.' | '1' | '2' | '3' | '4' | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd' | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|' ]

"""
        self._marker = marker

    def set_markeredgecolor(self, ec):
        """
Set the marker edge color

ACCEPTS: any matplotlib color - see help(colors)
"""
        self._markeredgecolor = ec

    def set_markeredgewidth(self, ew):
        """
Set the marker edge width in points

ACCEPTS: float value in points
"""
        self._markeredgewidth = ew

    def set_markerfacecolor(self, fc):
        """
Set the marker face color

ACCEPTS: any matplotlib color - see help(colors)
"""
        self._markerfacecolor = fc

    def set_markersize(self, sz):
        """
Set the marker size in points

ACCEPTS: float
"""
        self._markersize = sz

    def set_xdata(self, x):
        """
Set the data array for x

ACCEPTS: array
"""
        try: del self._xsorted
        except AttributeError: pass
            
        self.set_data(x, self._y)
        
    def set_ydata(self, y):
        """
Set the data array for y

ACCEPTS: array
"""

        self.set_data(self._x, y)
        
    def set_xclip(self, *args):
        """
Set the x clipping range for data clipping to xmin, xmax

ACCEPTS: (xmin, xmax)
"""
        if len(args)==1:
            xmin, xmax = args[0]
        else:
            xmin, xmax = args
        
        if xmax<xmin: xmax, xmin = xmin, xmax
        self._xmin, self._xmax = xmin, xmax
        self._set_clip()


        
    def set_yclip(self, *args):
        """
Set the y clipping range for data clipping to ymin, ymax

ACCEPTS: (ymin, ymax)
"""
        if len(args)==1:
            ymin, ymax = args[0]
        else:
            ymin, ymax = args
            
        if ymax<ymin: ymax, ymin = ymin, ymax
        self._ymin, self._ymax = ymin, ymax
        self._set_clip()

    def set_dashes(self, seq):
        """
Set the dash sequence, sequence of dashes with on off ink in points

ACCEPTS: sequence of on/off ink in points
"""
        if seq == (None, None):
            self.set_linestyle('-')
        else:
            self.set_linestyle('--')
        self._dashSeq = seq  # TODO: offset ignored for now
        
    def _draw_nothing(self, renderer, gc, xt, yt):
        pass

    def _draw_steps(self, renderer, gc, xt, yt):
        siz=len(xt)
        if siz<2: return
        xt2=ones((2*siz,), xt.typecode())
        xt2[0:-1:2], xt2[1:-1:2], xt2[-1]=xt, xt[1:], xt[-1]
        yt2=ones((2*siz,), yt.typecode())
        yt2[0:-1:2], yt2[1::2]=yt, yt
        gc.set_linestyle('solid')
        gc.set_capstyle('projecting')

        if self._newstyle:
            renderer.draw_lines(gc, xt2, yt2, self._transform)
        else:
            renderer.draw_lines(gc, xt2, yt2)

    def _draw_solid(self, renderer, gc, xt, yt):
        if len(xt)<2: return
        gc.set_linestyle('solid')
        gc.set_capstyle('projecting')   

        if self._newstyle:
            renderer.draw_lines(gc, xt, yt, self._transform)
        else:
            renderer.draw_lines(gc, xt, yt)


    def _draw_dashed(self, renderer, gc, xt, yt):
        if len(xt)<2: return
        gc.set_linestyle('dashed')
        if self._dashSeq is not None:
            gc.set_dashes(0, self._dashSeq)
        gc.set_capstyle('butt')   
        gc.set_joinstyle('miter') 

        if self._newstyle:
            renderer.draw_lines(gc, xt, yt, self._transform)
        else:
            renderer.draw_lines(gc, xt, yt)


    def _draw_dash_dot(self, renderer, gc, xt, yt):
        if len(xt)<2: return
        gc.set_linestyle('dashdot')
        gc.set_capstyle('butt')   
        gc.set_joinstyle('miter') 
        if self._newstyle:
            renderer.draw_lines(gc, xt, yt, self._transform)
        else:
            renderer.draw_lines(gc, xt, yt)



    def _draw_dotted(self, renderer, gc, xt, yt):

        if len(xt)<2: return
        gc.set_linestyle('dotted')
        gc.set_capstyle('butt')   
        gc.set_joinstyle('miter')
        if self._newstyle:
            renderer.draw_lines(gc, xt, yt, self._transform)
        else:
            renderer.draw_lines(gc, xt, yt)
        
    def _draw_point(self, renderer, gc, xt, yt):
        if self._newstyle:
            rgbFace = self._get_rgb_face()
            path = agg.path_storage()
            path.move_to(-0.5, -0.5)
            path.line_to(-0.5, 0.5)
            path.line_to(0.5, 0.5)
            path.line_to(0.5, -0.5)
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_arc(gc, None, x, y, 1, 1, 0.0, 360.0)

    def _draw_pixel(self, renderer, gc, xt, yt):
        if self._newstyle:
            rgbFace = self._get_rgb_face()            
            path = agg.path_storage()
            path.move_to(-0.5, -0.5)
            path.line_to(-0.5, 0.5)
            path.line_to(0.5, 0.5)
            path.line_to(0.5, -0.5)
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_point(gc, x, y)


    def _draw_circle(self, renderer, gc, xt, yt):

        w = h = renderer.points_to_pixels(self._markersize)
        
        rgbFace = self._get_rgb_face()

        if self._newstyle:
            N = 50.0
            r = w/2.
            rads = (2*math.pi/N)*arange(N)
            xs = r*cos(rads)
            ys = r*sin(rads)
            # todo: use curve3!
            path = agg.path_storage()
            path.move_to(xs[0], ys[0])
            for x, y in zip(xs[1:], ys[1:]):                
                path.line_to(x, y) 
            
            path.end_poly()
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_arc(gc, rgbFace,
                                  x, y, w, h, 0.0, 360.0)
        


    def _draw_triangle_up(self, renderer, gc, xt, yt):

        
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        rgbFace = self._get_rgb_face()

        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, offset)
            path.line_to(-offset, -offset)
            path.line_to(offset, -offset)
            path.end_poly()
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                verts = ( (x, y+offset),
                          (x-offset, y-offset),
                          (x+offset, y-offset) )
                renderer.draw_polygon(gc, rgbFace, verts)


    def _draw_triangle_down(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        rgbFace = self._get_rgb_face()

        if self._newstyle:
            
            path = agg.path_storage()
            path.move_to(-offset, offset)
            path.line_to(offset, offset)
            path.line_to(0, -offset)
            path.end_poly()
                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x-offset, y+offset),
                          (x+offset, y+offset),
                          (x, y-offset))
                renderer.draw_polygon(gc, rgbFace, verts)

    def _draw_triangle_left(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        rgbFace = self._get_rgb_face()

        if self._newstyle:
            
            path = agg.path_storage()
            path.move_to(-offset, 0)
            path.line_to(offset, -offset)
            path.line_to(offset, offset)
            path.end_poly()
                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x-offset, y),
                          (x+offset, y-offset),
                          (x+offset, y+offset))
                renderer.draw_polygon(gc, rgbFace, verts)


    def _draw_triangle_right(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        rgbFace = self._get_rgb_face()
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(offset, 0)
            path.line_to(-offset, -offset)
            path.line_to(-offset, offset)
            path.end_poly()
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x+offset, y),
                          (x-offset, y-offset),
                          (x-offset, y+offset))
                renderer.draw_polygon(gc, rgbFace, verts)



    def _draw_square(self, renderer, gc, xt, yt):
        side = renderer.points_to_pixels(self._markersize)
        offset = side*0.5
        rgbFace = self._get_rgb_face()
        
        if self._newstyle:
            
            path = agg.path_storage()
            path.move_to(-offset, -offset)
            path.line_to(-offset, offset)
            path.line_to(offset, offset)
            path.line_to(offset, -offset)
            path.end_poly()
                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:

            for (x,y) in zip(xt, yt):            
                renderer.draw_rectangle(
                    gc, rgbFace,
                    x-offset, y-offset, side, side) 

    def _draw_diamond(self, renderer, gc, xt, yt):
        offset = 0.6*renderer.points_to_pixels(self._markersize)
        rgbFace = self._get_rgb_face()
        if self._newstyle:            
            path = agg.path_storage()
            path.move_to(offset, 0)
            path.line_to(0, -offset)
            path.line_to(-offset, 0)
            path.line_to(0, offset)
            path.end_poly()
                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:


            for (x,y) in zip(xt, yt):            
                verts = ( (x+offset, y),
                          (x, y-offset),
                          (x-offset, y),
                          (x, y+offset))
                renderer.draw_polygon(gc, rgbFace, verts)

    def _draw_thin_diamond(self, renderer, gc, xt, yt):
        offset = 0.7*renderer.points_to_pixels(self._markersize)
        xoffset = 0.6*offset
        rgbFace = self._get_rgb_face()

        if self._newstyle:            
            path = agg.path_storage()
            path.move_to(xoffset, 0)
            path.line_to(0, -offset)
            path.line_to(-xoffset, 0)
            path.line_to(0, offset)
            path.end_poly()                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x+xoffset, y),
                          (x, y-offset),
                          (x-xoffset, y),
                          (x, y+offset))
                renderer.draw_polygon(gc, rgbFace, verts)

    def _draw_pentagon(self, renderer, gc, xt, yt):
        offset = 0.6*renderer.points_to_pixels(self._markersize)
        offsetX1 = offset*0.95
        offsetY1 = offset*0.31
        offsetX2 = offset*0.59
        offsetY2 = offset*0.81
        rgbFace = self._get_rgb_face()

        if self._newstyle:            
            path = agg.path_storage()
            path.move_to(0, offset)
            path.line_to(-offsetX1, offsetY1)
            path.line_to(-offsetX2, -offsetY2)
            path.line_to(+offsetX2, -offsetY2)
            path.line_to(+offsetX1, offsetY1)
            path.end_poly()
                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x, y+offset),
                          (x-offsetX1, y+offsetY1),
                          (x-offsetX2, y-offsetY2),
                          (x+offsetX2, y-offsetY2),
                          (x+offsetX1, y+offsetY1))
                renderer.draw_polygon(gc, rgbFace, verts)

    def _draw_hexagon1(self, renderer, gc, xt, yt):
        offset = 0.6*renderer.points_to_pixels(self._markersize)
        offsetX1 = offset*0.87
        offsetY1 = offset*0.5
        rgbFace = self._get_rgb_face()

        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, offset)
            path.line_to(-offsetX1, offsetY1)
            path.line_to(-offsetX1, -offsetY1)
            path.line_to(0, -offset)
            path.line_to(offsetX1, -offsetY1)
            path.line_to(offsetX1, offsetY1)
            path.end_poly()                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x, y+offset),
                          (x-offsetX1, y+offsetY1),
                          (x-offsetX1, y-offsetY1),
                          (x, y-offset),
                          (x+offsetX1, y-offsetY1),
                          (x+offsetX1, y+offsetY1))
                renderer.draw_polygon(gc, rgbFace, verts)

    def _draw_hexagon2(self, renderer, gc, xt, yt):
        offset = 0.6*renderer.points_to_pixels(self._markersize)
        offsetX1 = offset*0.5
        offsetY1 = offset*0.87
        rgbFace = self._get_rgb_face()
        if self._newstyle:            
            path = agg.path_storage()
            path.move_to(offset, 0)
            path.line_to(offsetX1, offsetY1)
            path.line_to(-offsetX1, offsetY1)
            path.line_to(-offset, 0)
            path.line_to(-offsetX1, -offsetY1)
            path.line_to(offsetX1, -offsetY1)
            path.end_poly()
                
            renderer.draw_markers(gc, path, rgbFace, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                verts = ( (x+offset, y),
                          (x+offsetX1, y+offsetY1),
                          (x-offsetX1, y+offsetY1),
                          (x-offset, y),
                          (x-offsetX1, y-offsetY1),
                          (x+offsetX1, y-offsetY1))
                renderer.draw_polygon(gc, rgbFace, verts)

    def _draw_vline(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, -offset)
            path.line_to(0, offset)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):            
                renderer.draw_line(gc, x, y-offset, x, y+offset)

    def _draw_hline(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(-offset, 0)
            path.line_to(offset, 0)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x-offset, y, x+offset, y)

    def _draw_tickleft(self, renderer, gc, xt, yt):
        offset = renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(-offset, 0)
            path.line_to(0, 0)                            
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x-offset, y, x, y)

    def _draw_tickright(self, renderer, gc, xt, yt):

        offset = renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, 0)
            path.line_to(offset, 0)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y, x+offset, y)

    def _draw_tickup(self, renderer, gc, xt, yt):
        offset = renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, 0)
            path.line_to(0, offset)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y, x, y+offset)

    def _draw_tickdown(self, renderer, gc, xt, yt):
        offset = renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, -offset)
            path.line_to(0, 0)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y-offset, x, y)

    def _draw_plus(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        if self._newstyle:
            
            path = agg.path_storage()
            path.move_to(-offset, 0)
            path.line_to( offset, 0)
            path.move_to( 0, -offset)
            path.line_to( 0, offset)
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x-offset, y, x+offset, y)
                renderer.draw_line(gc, x, y-offset, x, y+offset)

    def _draw_tri_down(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        offset1 = offset*0.8
        offset2 = offset*0.5
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, 0)
            path.line_to(0, -offset)
            path.move_to(0, 0)
            path.line_to(offset1, offset2)
            path.move_to(0, 0)
            path.line_to(-offset1, offset2)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y, x, y-offset)
                renderer.draw_line(gc, x, y, x+offset1, y+offset2)
                renderer.draw_line(gc, x, y, x-offset1, y+offset2)

    def _draw_tri_up(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        offset1 = offset*0.8
        offset2 = offset*0.5
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, 0)
            path.line_to(0, offset)
            path.move_to(0, 0)
            path.line_to(offset1, -offset2)
            path.move_to(0, 0)
            path.line_to(-offset1, -offset2)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y, x, y+offset)
                renderer.draw_line(gc, x, y, x+offset1, y-offset2)
                renderer.draw_line(gc, x, y, x-offset1, y-offset2)

    def _draw_tri_left(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        offset1 = offset*0.8
        offset2 = offset*0.5
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, 0)
            path.line_to(-offset, 0)
            path.move_to(0, 0)
            path.line_to(offset2, offset1)
            path.move_to(0, 0)
            path.line_to(offset2, -offset1)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y, x-offset, y)
                renderer.draw_line(gc, x, y, x+offset2, y+offset1)
                renderer.draw_line(gc, x, y, x+offset2, y-offset1)

    def _draw_tri_right(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        offset1 = offset*0.8
        offset2 = offset*0.5
        if self._newstyle:
            path = agg.path_storage()
            path.move_to(0, 0)
            path.line_to(offset, 0)
            path.move_to(0, 0)
            path.line_to(-offset2, offset1)
            path.move_to(0, 0)
            path.line_to(-offset2, -offset1)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x, y, x+offset, y)
                renderer.draw_line(gc, x, y, x-offset2, y+offset1)
                renderer.draw_line(gc, x, y, x-offset2, y-offset1)

    def _draw_x(self, renderer, gc, xt, yt):
        offset = 0.5*renderer.points_to_pixels(self._markersize)

        if self._newstyle:
            path = agg.path_storage()
            path.move_to(-offset, -offset)
            path.line_to(offset, offset)
            path.move_to(-offset, offset)
            path.line_to(offset, -offset)                
            renderer.draw_markers(gc, path, None, xt, yt, self._transform)
        else:
            for (x,y) in zip(xt, yt):
                renderer.draw_line(gc, x-offset, y-offset, x+offset, y+offset)
                renderer.draw_line(gc, x-offset, y+offset, x+offset, y-offset)

    def update_from(self, other):
        'copy properties from other to self'
        Artist.update_from(self, other)        
        self._linestyle = other._linestyle
        self._linewidth = other._linewidth
        self._color = other._color
        self._markersize = other._markersize        
        self._markerfacecolor = other._markerfacecolor
        self._markeredgecolor = other._markeredgecolor
        self._markeredgewidth = other._markeredgewidth
        self._dashSeq = other._dashSeq

        self._linestyle = other._linestyle
        self._marker = other._marker
        self._useDataClipping = other._useDataClipping

    def _get_rgb_face(self):
        if (self._markerfacecolor is None or
            (is_string_like(self._markerfacecolor) and
             self._markerfacecolor.lower()=='none') ): rgbFace = None
        else: rgbFace = colorConverter.to_rgb(self._markerfacecolor)
        return rgbFace

    # some aliases....
    def set_aa(self, val):
        'alias for set_antialiased'
        self.set_antialiased(val)
    
    def set_c(self, val):
        'alias for set_color'
        self.set_color(val)
    

    def set_ls(self, val):
        'alias for set_linestyle'
        self.set_linestyle(val)
    

    def set_lw(self, val):
        'alias for set_linewidth'
        self.set_linewidth(val)
    

    def set_mec(self, val):
        'alias for set_markeredgecolor'
        self.set_markeredgecolor(val)
    

    def set_mew(self, val):
        'alias for set_markeredgewidth'
        self.set_markeredgewidth(val)
    

    def set_mfc(self, val):
        'alias for set_markerfacecolor'
        self.set_markerfacecolor(val)
    

    def set_ms(self, val):
        'alias for set_markersize'
        self.set_markersize(val)
    
    def get_aa(self):
        'alias for get_antialiased'
        return self.get_antialiased()
    
    def get_c(self):
        'alias for get_color'
        return self.get_color()
    

    def get_ls(self):
        'alias for get_linestyle'
        return self.get_linestyle()
    

    def get_lw(self):
        'alias for get_linewidth'
        return self.get_linewidth()
    

    def get_mec(self):
        'alias for get_markeredgecolor'
        return self.get_markeredgecolor()
    

    def get_mew(self):
        'alias for get_markeredgewidth'
        return self.get_markeredgewidth()
    

    def get_mfc(self):
        'alias for get_markerfacecolor'
        return self.get_markerfacecolor()
    

    def get_ms(self):
        'alias for get_markersize'
        return self.get_markersize()
