"""
A Cairo backend for matplotlib implemented using pycairo
Author: Steve Chaplin
 
Cairo is a vector graphics library with cross-device output support.
Features of Cairo:
 * anti-aliasing
 * alpha channel 
 * in-memory image buffers
 * save image files:
   - PNG
   - PostScript (50% complete)
   - PDF        (proposed, 0% complete)

http://cairographics.org
http://www.freedesktop.org/Cairo/Home
Requires (in order, all available from Cairo website):
    libpixman, cairo, pycairo

cairo, pycairo from cvs fom Nov 06 2004 onwards are required

Naming Conventions
  * classes MixedUpperCase
  * varables lowerUpper
  * functions underscore_separated
"""

from __future__ import division

import os
import sys
def _fn_name(): return sys._getframe(1).f_code.co_name

from matplotlib import verbose
from matplotlib.numerix import asarray, pi, fromstring, UInt8 #, zeros, where,
     #transpose, nonzero, indices, ones, nx
#from matplotlib._matlab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import enumerate, True, False
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

import cairo

# add version checking, if cairo adds version number support
#version_required = (1,99,16)
#if gtk.pygtk_version < version_required:
#    raise SystemExit ("PyGTK %d.%d.%d is installed\n"
#                      "PyGTK %d.%d.%d or later is required"
#                      % (gtk.pygtk_version + version_required))
#backend_version = "%d.%d.%d" % gtk.pygtk_version
backend_version = 'unknown' # cairo does not report version, yet


DEBUG = False

# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 96

# Image formats that this backend supports - for print_figure()
IMAGE_FORMAT          = ['eps', 'png', 'ps', 'svg']
IMAGE_FORMAT_DEFAULT  = 'png'


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
    

    def __init__(self, surface, matrix, width, height, dpi):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        self.surface = surface
        self.matrix  = matrix
        self.width   = width
        self.height  = height
        self.dpi     = dpi
        self.text_ctx = cairo.Context()


    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        return self.width, self.height
    

    def get_text_scale(self):
        """
        Return the scale factor for fontsize taking screendpi and pixels per
        inch into account
        """
        return self.dpi.get()/PIXELS_PER_INCH


    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        if ismath:
            print 'ismath get_text_width_height() not implemented yet'
            return 1, 1
        else:
            ctx = self.text_ctx
            ctx.save()
            ctx.select_font (prop.get_name(),
                             self.fontangles [prop.get_style()],
                             self.fontweights[prop.get_weight()])
            scale = self.get_text_scale()
            size  = prop.get_size_in_points()
            # problem - scale remembers last setting and font can become
            # enormous causing program to crash
            # save/restore prevents the problem
            ctx.scale_font (scale*size)
        
            w, h = ctx.text_extents (s)[2:4]
            ctx.restore()
            
            return w, h

                              
    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0.
        If rgbFace is not None, fill the arc with it.
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # cairo draws circular arcs (width=height)
        # could curve_to() and draw a spline instead?
        radius = (height + width) / 4
        ctx    = gc.ctx
        ctx.new_path()
        ctx.arc (x, self.height - y, radius, angle1 * pi/180.0, angle2 * pi/180.0)

        if rgbFace:
            ctx.save()
            ctx.set_rgb_color (*rgbFace)
            ctx.fill()
            ctx.restore()
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
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()

        try: import cairo.numpy
        except:
            verbose.report_error("cairo.numpy module required for draw_image()")
            return

        # bbox - not used
        flipud = origin=='lower'

        ctx = cairo.Context()
        ctx.set_target_surface (self.surface)
        ctx.set_matrix (self.matrix)

        rows, cols, buffer = im.buffer_argb32()  # ARGB32, but colors still wrong
        X = fromstring(buffer, UInt8)
        X.shape = rows, cols, 4
        #print dir(im)
        #print 'im.get_size()', im.get_size()
        #print 'r,c', rows, cols

        # GTK method
        #rows, cols, s = im.as_str(flipud) # RGBA
        #print 'r,c', rows, cols
        #X = fromstring(s, UInt8)
        #X.shape = rows, cols, 4

        # ARGB32 
        surface = cairo.numpy.surface_create_for_array (X)

        # Alternative
        #surface = cairo.surface_create_for_image(buffer, cairo.FORMAT_ARGB32, cols, rows) #, stride)
        # error: TypeError: Cannot use string as modifiable buffer

        ctx.translate (x,y)
        ctx.show_surface (surface, cols, rows)


    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx
        ctx.new_path()
        ctx.move_to (x1, self.height - y1)
        ctx.line_to (x2, self.height - y2)
        ctx.stroke()


    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        y = [self.height - y for y in y]
        points = zip(x,y)
        x, y = points.pop(0)
        ctx = gc.ctx
        ctx.new_path()
        ctx.move_to (x, y)

        for x,y in points:
            ctx.line_to (x, y)
        ctx.stroke()


    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # render by drawing a 0.5 radius circle
        gc.ctx.new_path()
        gc.ctx.arc (x, self.height - y, 0.5, 0, 2*pi)
        gc.ctx.fill()


    def draw_polygon(self, gc, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex.
        If rgbFace is not None, fill the rectangle with it.
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()

        ctx = gc.ctx
        ctx.new_path()
        x, y = points[0]
        ctx.move_to (x, self.height - y)
        for x,y in points[1:]:
            ctx.line_to (x, self.height - y)
        ctx.close_path()

        if rgbFace:
            ctx.save()
            ctx.set_rgb_color (*rgbFace)
            ctx.fill()
            ctx.restore()
        ctx.stroke()


    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        """
        Draw a non-filled rectangle at x,y (lower left) with width and height,
        using the GraphicsContext gcEdge.
        Draw a filled rectangle within it of color rgbFace, if rgbFace is not
        None.
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx
        ctx.new_path()
        ctx.rectangle (x, self.height - y - height, width, height)
        if rgbFace:
            ctx.save()
            ctx.set_rgb_color (*rgbFace)
            ctx.fill()
            ctx.restore()
        ctx.stroke()


    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):    
        """
        Render the matplotlib.text.Text instance at x, y in window
        coords using GraphicsContext gc
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx

        if ismath:
            verbose.report_error('Mathtext not implemented yet')
        else:
            # text is looking too small - size, scale problem?
            ctx.new_path()
            ctx.move_to (x, y)
            ctx.select_font (prop.get_name(),
                             self.fontangles [prop.get_style()],
                             self.fontweights[prop.get_weight()])
            scale = self.get_text_scale()
            size  = prop.get_size_in_points()

            ctx.save()
            if angle:
                ctx.rotate (-angle * pi / 180)
            ctx.scale_font (scale*size)
            ctx.show_text (s)
            ctx.restore()

         
    def flipy(self):
        """return true if y small numbers are top for renderer"""
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        return True

    
    def new_gc(self):
        """
        Return an instance of a GraphicsContextCairo
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        gc = GraphicsContextCairo (renderer=self)
        gc.ctx.set_target_surface (self.surface)
        gc.ctx.set_matrix (self.matrix)
        return gc


    def points_to_pixels(self, points):
        """
        Convert points to display units (as a float).
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
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
        'projecting' : cairo.LINE_CAP_SQUARE,
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
            verbose.report_error('Unrecognized cap style.  Found %s' % cs)


    def set_clip_rectangle(self, rectangle):
        """
        Set the clip rectangle with sequence (left, bottom, width, height)
        """
        # Cairo clipping is currently extremely slow
        # cairo/BUGS lists it as a known bug
        self._cliprect = rectangle

        x,y,w,h = rectangle
        ctx = self.ctx
        ctx.new_path()
        ctx.rectangle (x, self.renderer.height - h - y, w, h)

        #ctx.save()     # uncomment to view the clip rectangle
        #ctx.set_rgb_color(1,0,0)
        #ctx.set_line_width(6)
        #ctx.stroke()
        #ctx.restore()        

        ctx.clip ()
        

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes == None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            dashes_pixels = self.renderer.points_to_pixels(asarray(dashes))
            self.ctx.set_dash(dashes_pixels, offset)
        

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
            verbose.report_error('Unrecognized join style.  Found %s' % js)


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
    if DEBUG: print 'backend_cairo.%s()' % _fn_name()
    pass


def show():
    """
    This is usually the last line of a matlab script and tells the
    backend that it is time to draw.  In interactive mode, this may be
    a do nothing func.  See the GTK backend for an example of how to
    handle interactive versus batch mode
    """
    if DEBUG: print 'backend_cairo.%s()' % _fn_name()
    for manager in Gcf.get_all_fig_managers():
        manager.canvas.realize()


def new_figure_manager(num, *args, **kwargs): # called by backends/__init__.py
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_cairo.%s()' % _fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas  = FigureCanvasCairo(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


def print_figure_fn(figure, filename, dpi=150, facecolor='w', edgecolor='w',
                    orientation='portrait'):
    """
    Render the figure to hardcopy.  Set the figure patch face and
    edge colors.  This is useful because some of the GUIs have a
    gray figure face color background and you'll probably want to
    override this on hardcopy

    orientation - only currently applies to PostScript printing.
    filename - can also be a file object, png format is assumed
    """
    if DEBUG: print 'backend_cairo.FigureCanvasCairo.%s()' % _fn_name()

    # save figure state
    origDPI       = figure.dpi.get()
    origfacecolor = figure.get_facecolor()
    origedgecolor = figure.get_edgecolor()
        
    # settings for printing
    figure.dpi.set(dpi)
    figure.set_facecolor(facecolor)
    figure.set_edgecolor(edgecolor)        

    if isinstance(filename, file):   # eg when do savefig(sys.stdout)
        _save_png (figure, filename) # assume PNG format
    else:
        root, ext = os.path.splitext(filename)       
        ext = ext[1:]
        if ext == '':
            ext      = IMAGE_FORMAT_DEFAULT
            filename = filename + '.' + ext

        ext = ext.lower()
        #if ext in ('png', 'ps'):  # native formats
        if ext in ('png'):  # native formats
            try:
                fileObject = file(filename,'wb')
            except IOError, exc:
                verbose.report_error("%s: %s" % (exc.filename, exc.strerror))
            else:
                if ext == 'png': _save_png (figure, fileObject)
                else:            _save_ps  (figure, fileObject, orientation)
            
        #elif ext in ('eps', 'svg'): # backend_svg/ps
        elif ext in ('eps', 'ps', 'svg'): # backend_svg/ps
            #if ext == 'eps':
            if ext in ('eps', 'ps'):
                from backend_ps import FigureCanvasPS  as FigureCanvas
            else:
                from backend_svg import FigureCanvasSVG as FigureCanvas
            fc = FigureCanvas(figure)
            fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)

        else:
            verbose.report_error('Format "%s" is not supported.\nSupported formats: %s.' %
                                 (ext, ', '.join(IMAGE_FORMAT)))

    # restore the new params
    figure.dpi.set(origDPI)
    figure.set_facecolor(origfacecolor)
    figure.set_edgecolor(origedgecolor)
        
        
def _save_png (figure, fileObject):
    width, height = figure.get_width_height()
    width, height = int(width), int(height)

    ctx = cairo.Context()
    # 4 png formats supported
    ctx.set_target_png (fileObject, cairo.FORMAT_ARGB32, width, height)
    renderer = RendererCairo (ctx.target_surface, ctx.matrix,
                              width, height, figure.dpi)
    figure.draw(renderer)
    ctx.show_page()
        

def _save_ps (figure, fileObject, orientation):
    # Cairo produces PostScript Level 3
    # 'ggv' can't read cairo ps files, but 'gv' can

    # not currently working properly - the figure comes out the wrong size
    ppi = 300.0
    figure.dpi.set(72)

    w_in, h_in = figure.get_size_inches()
    width, height = figure.get_width_height()
    
    ctx = cairo.Context()
    ctx.set_target_ps (fileObject, w_in, h_in, ppi, ppi)

    orientation = 'portrait' # landscape not supported yet
    if orientation == 'landscape':
        # cairo/src/cairo_ps_surface.c
        # '%%Orientation: Portrait' is always written to the file header
        # '%%Orientation: Landscape' would possibly cause problems
        # since some printers would rotate again ?
        # TODO:
        # 1) needs -pi/2 rotation, centered (and maximised?)
        #    don't know how to rotate without text being distorted
        # 2) add portrait/landscape checkbox to FileChooser
        pass
        
    renderer = RendererCairo (ctx.target_surface, ctx.matrix, width, height, figure.dpi)
            
    show_fig_border = True  # for testing figure orientation and scaling
    if show_fig_border:
        ctx.new_path()
        ctx.rectangle(0, 0, width, height)
        ctx.set_line_width(4.0)
        ctx.set_rgb_color(1,0,0)
        ctx.stroke()
        ctx.new_path()
        ctx.rectangle(0+20, 0+20, width-20, height-20)
        ctx.set_line_width(4.0)
        ctx.set_rgb_color(1,0,0)
        ctx.stroke()
        ctx.move_to(30,30)
        ctx.select_font('sans-serif')
        ctx.scale_font(20)
        ctx.show_text('Origin corner')
    ctx.show_page()


class FigureCanvasCairo(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...
    Public attributes
      figure - A Figure instance
    """
    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):
        print_figure_fn(self.figure, filename, dpi, facecolor, edgecolor,
                      orientation)
