"""
A Cairo backend for matplotlib implemented using pycairo
Author: Steve Chaplin
 
http://www.freedesktop.org/Cairo/Home
http://cairographics.org
Requires (in order, all available from Cairo website):
    libpixman, cairo, libsvg, libsvg-cairo, pycairo

Cairo is a vector graphics library with cross-device output support.
Features of Cairo:
 * anti-aliasing
 * alpha channel 
 * in-memory image buffers
 * PNG
 * PostScript
 * PDF (in development)
 * SVG (in development)

Naming Conventions
  * classes MixedUpperCase
  * varables lowerUpper
  * functions underscore_separated
"""

import sys
def _fn_name(): return sys._getframe(1).f_code.co_name

from matplotlib import verbose
from matplotlib.numerix import asarray, pi #, fromstring, UInt8, zeros, where, transpose, nonzero, indices, ones, nxfrom matplotlib._matlab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase, error_msg
from matplotlib.cbook import enumerate, True, False
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

try:
    import cairo
    # version > x, check - later
except:
    verbose.report_error('PyCairo is required to run the Matplotlib Cairo backend')
    raise SystemExit()
backend_version = '0.1.23' # cairo does not report version, yet


Debug = False


# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 96


def error_msg_cairo(msg, *args):
    """
    Signal an error condition -- in a GUI, popup a error dialog
    """
    if Debug: print 'backend_cairo.%s()' % _fn_name()
    verbose.report('Error: %s'%msg)
    sys.exit()

    
class RendererCairo(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """
    fontweights = {
        100          : cairo.FONT_WEIGHT_NORMAL,
        200          : cairo.FONT_WEIGHT_NORMAL,
        300          : cairo.FONT_WEIGHT_NORMAL,
        400          : cairo.FONT_WEIGHT_NORMAL,
        500          : cairo.FONT_WEIGHT_NORMAL,
        600          : cairo.FONT_WEIGHT_BOLD,
        700          : cairo.FONT_WEIGHT_BOLD,
        800          : cairo.FONT_WEIGHT_BOLD,
        900          : cairo.FONT_WEIGHT_BOLD,
        'ultralight' : cairo.FONT_WEIGHT_NORMAL,
        'light'      : cairo.FONT_WEIGHT_NORMAL,
        'normal'     : cairo.FONT_WEIGHT_NORMAL,
        'medium'     : cairo.FONT_WEIGHT_NORMAL,
        'semibold'   : cairo.FONT_WEIGHT_BOLD,
        'bold'       : cairo.FONT_WEIGHT_BOLD,
        'heavy'      : cairo.FONT_WEIGHT_BOLD,
        'ultrabold'  : cairo.FONT_WEIGHT_BOLD,
        'black'      : cairo.FONT_WEIGHT_BOLD,
                   }
    fontangles = {
        'italic'  : cairo.FONT_SLANT_ITALIC,
        'normal'  : cairo.FONT_SLANT_NORMAL,
        'oblique' : cairo.FONT_SLANT_OBLIQUE,
        }
    

    def __init__(self, surface, width, height, dpi):
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        self.surface = surface
        self.width   = width
        self.height  = height
        self.dpi     = dpi    # should not need dpi? Cairo is device independent


    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        return self.width, self.height
    

    def get_text_scale(self):
        """
        Return the scale factor for fontsize taking screendpi and pixels per
        inch into account
        """
        # copied from backend_gtk, is this needed for Cairo?
        return self.dpi.get()/PIXELS_PER_INCH


    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # if ismath:  # do later

        ctx = cairo.Context() # later - save a ctx specifically for this op?
        ctx.select_font (prop.get_name(),
                         self.fontangles [prop.get_style()],
                         self.fontweights[prop.get_weight()])
        scale = self.get_text_scale()
        size  = prop.get_size_in_points()
        ctx.scale_font (scale*size)
        
        w, h = ctx.text_extents (s)[2:4]
        return w, h

                              
    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0.

        If rgbFace is not None, fill the arc with it.  gcEdge
        is a GraphicsContext instance
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # cairo draws circular arcs (width=height) only?
        ctx = gcEdge.ctx
        y = self.height - y
        radius = width / 2

        if rgbFace:
            color_save = ctx.rgb_color
            ctx.set_rgb_color (*rgbFace)
            ctx.arc (x, y, radius, angle1 * pi/180.0, angle2 * pi/180.0)
            ctx.fill()
            ctx.set_rgb_color (*color_save)
            
        ctx.arc (x, y, radius, angle1 * pi/180.0, angle2 * pi/180.0)
        ctx.stroke()
    
    
    def draw_image(self, x, y, im, origin, bbox):
        """
        Draw the Image instance into the current axes; x is the
        distance in pixels from the left hand side of the canvas. y is
        the distance from the origin.  That is, if origin is upper, y
        is the distance from top.  If origin is lower, y is the
        distance from bottom

        origin is 'upper' or 'lower'

        bbox is a matplotlib.transforms.BBox instance for clipping, or
        None
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        pass
        # how does cairo do this?
    

    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx
        y1 = self.height - y1
        y2 = self.height - y2
        ctx.move_to (x1, y1); ctx.line_to (x2, y2)
        ctx.stroke()


    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx
        y = [self.height - y for y in y]
        points = zip(x,y)
        x, y = points.pop(0)
        ctx.move_to (x, y)

        for x,y in points:
            ctx.line_to (x, y)
        ctx.stroke()


    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # render by drawing a 0.5 radius circle
        y = self.height - y
        ctx.arc (x, y, 0.5, 0, 2*pi)
        ctx.fill()


    def draw_polygon(self, gcEdge, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """  
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gcEdge.ctx
        points = [(x, (self.height-y)) for x,y in points]

        if rgbFace:
            color_save = ctx.rgb_color
            ctx.set_rgb_color (*rgbFace)
            ctx.new_path()
            x, y = points[0]
            ctx.move_to (x, y)
            for x,y in points[1:]:
                ctx.line_to (x, y)
            ctx.close_path ()
            ctx.fill()
            ctx.set_rgb_color (*color_save)

        ctx.new_path()
        x, y = points[0]
        ctx.move_to (x, y)
        for x,y in points[1:]:
            ctx.line_to (x, y)
        ctx.close_path ()
        ctx.stroke()


    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        """
        Draw a non-filled rectangle at x,y (lower left) with width and height,
        using the GraphicsContext gcEdge.
        Draw a filled rectangle within it of color rgbFace, if rgbFace is not
        None.
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gcEdge.ctx
        y = self.height - y - height

        if rgbFace:
            color_save = ctx.rgb_color
            ctx.set_rgb_color (*rgbFace)
            ctx.rectangle (x, y, width, height)
            ctx.fill()
            ctx.set_rgb_color (*color_save)

        ctx.rectangle (x, y, width, height)
        ctx.stroke()


    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):    
        """
        Render the matplotlib.text.Text instance at x, y in window
        coords using GraphicsContext gc
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx

        if ismath:
            verbose.report_error('Mathtext not implemented yet')
            #self._draw_mathtext(gc, x, y, s, prop, angle)
        else:
            # see also get_text_width_height()
            # text is looking too small - size, scale problem?
            ctx.select_font (prop.get_name(),
                             self.fontangles [prop.get_style()],
                             self.fontweights[prop.get_weight()])
            scale = self.get_text_scale()
            size  = prop.get_size_in_points()
            ctx.scale_font (scale*size)
            ctx.move_to (x, y)
            if angle: # rotated text looks awful!
                ctx.rotate (-angle * pi / 180)
            ctx.show_text (s)

         
    def flipy(self):
        """return true if y small numbers are top for renderer"""
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        return True

    
    def new_gc(self):
        """
        Return an instance of a GraphicsContextCairo
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        gc = GraphicsContextCairo (renderer=self)
        if self.surface:
            gc.ctx.set_target_surface (self.surface)
        #else: # testing
        #    gc.ctx.set_target_png (file, cairo.FORMAT_ARGB32, self.width, self.height)
        #    set_target_png(), set_target_ps() are not wrapped
        return gc


    def points_to_pixels(self, points):
        """
        Convert points to display units.
        """
        if Debug: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # should not need points_to_pixels() ? Cairo is device independent
        return points * PIXELS_PER_INCH/72.0 * self.dpi.get()/72.0


class GraphicsContextCairo(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc...
    """
    _joind = {
        'bevel' : cairo.LINE_JOIN_BEVEL,
        'miter' : cairo.LINE_JOIN_MITER,
        'round' : cairo.LINE_JOIN_ROUND,
        }

    _capd = {
        'butt'       : cairo.LINE_CAP_BUTT,
        'projecting' : cairo.LINE_CAP_SQUARE,  # same as CAP_PROJECTING ?
        'round'      : cairo.LINE_CAP_ROUND,
        }

    
    def __init__(self, renderer):
        GraphicsContextBase.__init__(self)
        self.renderer = renderer
        self.ctx = cairo.Context()

        
    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending
        """
        self._alpha = alpha
        self.ctx.set_alpha(alpha)


    def set_capstyle(self, cs):
        """
        Set the capstyle as a string in ('butt', 'round', 'projecting')
        """
        if cs in ('butt', 'round', 'projecting'):
            self._capstyle = cs
            self.ctx.set_line_cap (self._capd[cs])
        else:
            error_msg('Unrecognized cap style.  Found %s' % cs)


    def set_clip_rectangle(self, rectangle):
        """
        Set the clip rectangle with sequence (left, bottom, width, height)
        """
        self._cliprect = rectangle
        ctx = self.ctx

        x,y,w,h = rectangle
        widget_h = self.renderer.height
        # 1 clipping is extremely slow so I disabled it (its due to be optimised?)
        # 2 how to specify a clip rect w/o actually drawing it!

        #ctx.new_path()
        #ctx.move_to (x-1,   widget_h-y)
        #ctx.line_to (x-10,   widget_h-y-h)
        #ctx.line_to (x-10+w, widget_h-y-h)
        #ctx.line_to (x-10+w, widget_h-y)
        #ctx.line_to (x-10,   widget_h-y)
        #ctx.close_path()
        
        #ctx.clip ()
        

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if offset != None:
            dashes_pixels = self.renderer.points_to_pixels(asarray(dashes))
            self.ctx.set_dash(dashes_pixels, offset)
        else: # hack to switch dashes off
            self.ctx.set_dash([1,0], 0)
        

    def set_foreground(self, fg, isRGB=None):
        """
        Set the foreground color.  fg can be a matlab format string, a
        html hex color string, an rgb unit tuple, or a float between 0
        and 1.  In the latter case, grayscale is used.
        """
        GraphicsContextBase.set_foreground(self, fg, isRGB)
        self.ctx.set_rgb_color(*self._rgb)


    def set_joinstyle(self, js):
        """
        Set the join style to be one of ('miter', 'round', 'bevel')
        """
        if js in ('miter', 'round', 'bevel'):
            self._joinstyle = js
            self.ctx.set_line_join(self._joind[js])
        else:
            error_msg('Unrecognized join style.  Found %s' % js)


    def set_linewidth(self, w):
        """
        Set the linewidth in points
        """
        self._linewidth = w
        self.ctx.set_line_width (self.renderer.points_to_pixels(w))

        
########################################################################
#    
# The following functions and classes are for matlab compatibility
# mode (matplotlib.matlab) and implement window/figure managers,
# etc...
#
########################################################################

def draw_if_interactive():
    """
    This should be overriden in a windowing environment if drawing
    should be done in interactive python mode
    """
    if Debug: print 'backend_cairo.%s()' % _fn_name()
    pass


def show():
    """
    This is usually the last line of a matlab script and tells the
    backend that it is time to draw.  In interactive mode, this may be
    a do nothing func.  See the GTK backend for an example of how to
    handle interactive versus batch mode
    """
    if Debug: print 'backend_cairo.%s()' % _fn_name()
    for manager in Gcf.get_all_fig_managers():
        manager.canvas.realize()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if Debug: print 'backend_cairo.%s()' % _fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasCairo(thisFig)
    manager = FigureManagerCairo(canvas, num)
    return manager


class FigureCanvasCairo(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
    """

    def draw(self):
        """
        Draw the figure using the renderer
        """
        if Debug: print 'backend_cairo.FigureCanvasCairo%s()' % _fn_name()
        pass
        #renderer = RendererCairo()  # height, width, figure.dpi
        #self.figure.draw(renderer)
        
        
    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w', orientation='portrait'):
        """
        Render the figure to hardcopy.  Set the figure patch face and
        edge colors.  This is useful because some of the GUIs have a
        gray figure face color background and you'll probably want to
        override this on hardcopy
        """
        if Debug: print 'backend_cairo.FigureCanvasCairo%s()' % _fn_name()
        # set the new parameters
        origDPI       = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        
        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)        


        l,b,width, height = self.figure.bbox.get_bounds()
        renderer = RendererCairo (None, height, width, self.figure.dpi)
        #ctx.set_target_png ( FILE *file, cairo_format_t format, width, height) - not wrapped?

        # 1 draw to an image, -> save image as a file
        # 2 or draw direct to a (png) file
        
        self.figure.draw(renderer)
        # do something to save to hardcopy

        print 'saving hardcopy, not implemented yet'

        # restore the new params and redraw the screen if necessary
        self.figure.dpi.set(origDPI)
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.draw()
        
        
    def realize(self, *args): # is not required in cairo?
        """
        This method will be called when the system is ready to draw,
        eg when a GUI window is realized
        """
        if Debug: print 'backend_cairo.FigureCanvasCairo%s()' % _fn_name()
        self._isRealized = True  
        self.draw()

    
class FigureManagerCairo(FigureManagerBase):
    """
    Wrap everything up into a window for the matlab interface

    For non interactive backends, the base class does all the work
    """
    pass


########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################


FigureManager = FigureManagerCairo
error_msg = error_msg_cairo
