"""
A Cairo backend for matplotlib
Author: Steve Chaplin
 
Cairo is a vector graphics library with cross-device output support.
Features of Cairo:
 * anti-aliasing
 * alpha channel 
 * saves image files as PNG, PostScript, PDF

http://cairographics.org
Requires (in order, all available from Cairo website):
    libpixman, cairo, pycairo

Naming Conventions
  * classes MixedUpperCase
  * varables lowerUpper
  * functions underscore_separated
"""

from __future__ import division
import os
import sys
import warnings
def _fn_name(): return sys._getframe(1).f_code.co_name

import matplotlib.agg as agg
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook      import enumerate, izip
from matplotlib.figure     import Figure
from matplotlib.mathtext   import math_parse_s_ft2font
import matplotlib.numerix as numx
from matplotlib.transforms import Bbox

import cairo

_version_required = (0,5,0)
if cairo.version_info < _version_required:
   raise SystemExit ("Pycairo %d.%d.%d is installed\n"
                     "Pycairo %d.%d.%d or later is required"
                     % (cairo.version_info + _version_required))
backend_version = cairo.version
del _version_required

if hasattr (cairo.ImageSurface, 'create_for_array'):
   HAVE_CAIRO_NUMPY = True
else:
   HAVE_CAIRO_NUMPY = False


_debug = False
#_debug = True

# Image formats that this backend supports - for print_figure()
IMAGE_FORMAT          = ['eps', 'png', 'ps', 'svg']
#IMAGE_FORMAT          = ['eps', 'pdf', 'png', 'ps', 'svg'] # pdf not ready yet
IMAGE_FORMAT_DEFAULT  = 'png'


class RendererCairo(RendererBase):
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
    

    def __init__(self, dpi):
        """width, height - the canvas width, height. Is not necessarily
        the same as the surface (pixmap) width, height
        """
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        self.dpi      = dpi
        surface = cairo.ImageSurface (cairo.FORMAT_ARGB32,1,1)
        self.text_ctx = cairo.Context (surface)

    def _set_pixmap(self, pixmap):
        self.surface = pixmap

    def _set_width_height(self, width, height):
        self.width  = width
        self.height = height
        self.matrix_flipy = cairo.Matrix (yy=-1, y0=self.height)
        # use matrix_flipy for ALL rendering?
        # - problem with text? - will need to switch matrix_flipy off, or do a
        # font transform?

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        # draws circular arcs where width=height
        # FIXME
        # to get a proper arc of width/height you can use translate() and
        # scale(), see draw_arc() manual page
        radius = (height + width) / 4
        ctx    = gc.ctx
        ctx.new_path()
        ctx.arc (x, self.height - y, radius,
                 angle1 * numx.pi/180.0, angle2 * numx.pi/180.0)

        if rgbFace:
            ctx.save()
            ctx.set_source_rgb (*rgbFace)
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()
    
    
    def draw_image(self, x, y, im, origin, bbox):
        # bbox - not currently used
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        if numx.which[0] == "numarray":
            warnings.warn("draw_image() currently works for numpy, but not "
                          "numarray")
            return

        if not HAVE_CAIRO_NUMPY:
            warnings.warn("cairo with Numeric support is required for "
                          "draw_image()")
            return

        #flipud = origin=='lower'             # gtk method (uses RGBA)
        #rows, cols, s = im.as_str(flipud)
        rows, cols, buf = im.buffer_argb32()  # ARGB32, but colors still wrong
        X = numx.fromstring (buf, numx.UInt8)
        X.shape = rows, cols, 4

        # function does not pass a 'gc' so we create our own context
        ctx = cairo.Context (self.surface)
        surface = cairo.ImageSurface.create_for_array (X)
        ctx.set_source_surface (surface, x, y)
        ctx.paint()
        

    def draw_line(self, gc, x1, y1, x2, y2):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        ctx = gc.ctx
        ctx.new_path()
        ctx.move_to (x1, self.height - y1)
        ctx.line_to (x2, self.height - y2)
        ctx.stroke()


    def draw_lines(self, gc, x, y, transform=None):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        if transform:
            if transform.need_nonlinear():
                x, y = transform.nonlinear_only_numerix(x, y)
            x, y = transform.numerix_x_y(x, y)
        
        ctx = gc.ctx
        matrix_old = ctx.get_matrix()
        ctx.set_matrix (self.matrix_flipy)

        points = izip(x,y)
        x, y = points.next()
        ctx.new_path()
        ctx.move_to (x, y)

        for x,y in points:
            ctx.line_to (x, y)
        ctx.stroke()

        ctx.set_matrix (matrix_old)


    def draw_markers(self, gc, path, rgbFace, x, y, transform):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        ctx = gc.ctx

        if transform.need_nonlinear():
            x,y = transform.nonlinear_only_numerix(x, y)
        
        x, y = transform.numerix_x_y(x, y) # do nonlinear and affine transform

        # TODO - use cairo transform
        # matrix worked for dotted lines, but not markers in line_styles.py
        # it upsets/transforms generate_path() ?
        # need to flip y too, and update generate_path() ?
        # the a,b,c,d,tx,ty affine which transforms x and y
        #vec6 = transform.as_vec6_val() # not used (yet)
        #matrix_old = ctx.get_matrix()
        #ctx.set_matrix (cairo.Matrix (*vec6))

        path_list = [path.vertex() for i in range(path.total_vertices())]

        def generate_path (path_list):
           for code, xp, yp in path_list:
               if code == agg.path_cmd_move_to:
                  ctx.move_to (xp, -yp)
               elif code == agg.path_cmd_line_to:
                  ctx.line_to (xp, -yp)
               elif code == agg.path_cmd_end_poly:
                  ctx.close_path()

        for x,y in izip(x,y):
            ctx.save()
            ctx.new_path()
            ctx.translate(x, self.height - y)
            generate_path (path_list)

            if rgbFace:
               ctx.save()
               ctx.set_source_rgb (*rgbFace)
               # later - set alpha also?                     
               ctx.fill_preserve()
               ctx.restore() # undo colour change and restore path
            
            ctx.stroke()
            ctx.restore() # undo translate()

        #ctx.set_matrix(matrix_old)

        
    def draw_point(self, gc, x, y):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        # render by drawing a 0.5 radius circle
        gc.ctx.new_path()
        gc.ctx.arc (x, self.height - y, 0.5, 0, 2*numx.pi)
        gc.ctx.fill()


    def draw_polygon(self, gc, rgbFace, points):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        ctx = gc.ctx
        matrix_old = ctx.get_matrix()
        ctx.set_matrix (self.matrix_flipy)
        
        ctx.new_path()
        x, y = points[0]
        ctx.move_to (x, y)
        for x,y in points[1:]:
            ctx.line_to (x, y)
        ctx.close_path()

        if rgbFace:
            ctx.save()
            ctx.set_source_rgb (*rgbFace)
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()
        
        ctx.set_matrix (matrix_old)

    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        ctx = gc.ctx
        ctx.new_path()
        ctx.rectangle (x, self.height - y - height, width, height)
        if rgbFace:
            ctx.save()
            ctx.set_source_rgb (*rgbFace)
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()


    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        # Note: x,y are device/display coords, not user-coords, unlike other
        # draw_* methods
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        if ismath:
           self._draw_mathtext(gc, x, y, s, prop, angle)

        else:
           ctx = gc.ctx
           ctx.new_path()
           ctx.move_to (x, y)
           ctx.select_font_face (prop.get_name(),
                                 self.fontangles [prop.get_style()],
                                 self.fontweights[prop.get_weight()])

           # size = prop.get_size_in_points() * self.dpi.get() / 96.0
           size = prop.get_size_in_points() * self.dpi.get() / 72.0
        
           ctx.save()
           if angle:
              ctx.rotate (-angle * numx.pi / 180)
           ctx.set_font_size (size)
           ctx.show_text (s)
           ctx.restore()


    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
       # mathtext using the gtk/gdk method

        if numx.which[0] == "numarray":
            warnings.warn("_draw_mathtext() currently works for numpy, but "
                          "not numarray")
            return

        if not HAVE_CAIRO_NUMPY:
            warnings.warn("cairo with Numeric support is required for "
                          "_draw_mathtext()")
            return

        size = prop.get_size_in_points()
        width, height, fonts = math_parse_s_ft2font(
            s, self.dpi.get(), size)

        if angle==90:
            width, height = height, width
            x -= width
        y -= height

        imw, imh, s = fonts[0].image_as_str()
        N = imw*imh

        # a numpixels by num fonts array
        Xall = numx.zeros((N,len(fonts)), typecode=numx.UInt8)

        for i, font in enumerate(fonts):
            if angle == 90:
                font.horiz_image_to_vert_image() # <-- Rotate
            imw, imh, s = font.image_as_str()
            Xall[:,i] = numx.fromstring(s, numx.UInt8)  

        # get the max alpha at each pixel
        Xs = numx.mlab.max (Xall,1)

        # convert it to it's proper shape
        Xs.shape = imh, imw

        pa = numx.zeros(shape=(imh,imw,4), typecode=numx.UInt8)
        rgb = gc.get_rgb()
        pa[:,:,0] = int(rgb[0]*255)
        pa[:,:,1] = int(rgb[1]*255)
        pa[:,:,2] = int(rgb[2]*255)
        pa[:,:,3] = Xs

        # works for numpy pa, not a numarray pa
        surface = cairo.ImageSurface.create_for_array (pa)
        gc.ctx.set_source_surface (surface, x, y)
        gc.ctx.paint()
        #gc.ctx.show_surface (surface, imw, imh)
            

    def flipy(self):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        return True
        #return False # tried - all draw objects ok except text (and images?)
        # which comes out mirrored!

    
    def get_canvas_width_height(self):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        return self.width, self.height
    

    def get_text_width_height(self, s, prop, ismath):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        if ismath:
            width, height, fonts = math_parse_s_ft2font(
               s, self.dpi.get(), prop.get_size_in_points())
            return width, height

        ctx = self.text_ctx
        ctx.save()
        ctx.select_font_face (prop.get_name(),
                              self.fontangles [prop.get_style()],
                              self.fontweights[prop.get_weight()])

        # Cairo (says it) uses 1/96 inch user space units, ref: cairo_gstate.c
        # but if /96.0 is used the font is too small

        #size = prop.get_size_in_points() * self.dpi.get() / 96.0
        size = prop.get_size_in_points() * self.dpi.get() / 72.0
        
        # problem - scale remembers last setting and font can become
        # enormous causing program to crash
        # save/restore prevents the problem
        ctx.set_font_size (size)
        
        w, h = ctx.text_extents (s)[2:4]
        ctx.restore()
            
        return w, h

                              
    def new_gc(self):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        gc = GraphicsContextCairo (renderer=self, surface=self.surface)
        return gc


    def points_to_pixels(self, points):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        return points/72.0 * self.dpi.get()
     

class GraphicsContextCairo(GraphicsContextBase):
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

    
    def __init__(self, renderer, surface):
        GraphicsContextBase.__init__(self)
        self.renderer = renderer
        self.ctx = cairo.Context (surface)
        
    def set_alpha(self, alpha):
        self._alpha = alpha
        rgb = self._rgb
        self.ctx.set_source_rgba (rgb[0], rgb[1], rgb[2], alpha)

    #def set_antialiased(self, b):
        # enable/disable anti-aliasing is not (yet) supported by Cairo


    def set_capstyle(self, cs):
        if cs in ('butt', 'round', 'projecting'):
            self._capstyle = cs
            self.ctx.set_line_cap (self._capd[cs])
        else:
            raise ValueError('Unrecognized cap style.  Found %s' % cs)


    def set_clip_rectangle(self, rectangle):
        # Cairo < 0.4.0: clipping is currently extremely slow
        # Cairo 0.4.0  : pixel-aligned rectangular clip-regions are now faster
        self._cliprect = rectangle

        x,y,w,h = rectangle
        # pixel-aligned clip-regions are faster        
        x,y,w,h = round(x), round(y), round(w), round(h)
        ctx = self.ctx
        ctx.new_path()
        ctx.rectangle (x, self.renderer.height - h - y, w, h)

        #ctx.save()     # uncomment to view the clip rectangle
        #ctx.set_source_rgb(1,0,0)
        #ctx.set_line_width(6)
        #ctx.stroke()
        #ctx.restore()        

        #ctx.init_clip() # not needed? used unsuccessfully to fix clip problem
        # when uncomment ctx.clip() it causes problems in line_styles.py
        # - multiple axes, only the first one has its background drawn
        #ctx.clip ()
        

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes == None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            self.ctx.set_dash (
               self.renderer.points_to_pixels (numx.asarray(dashes)), offset)
        

    def set_foreground(self, fg, isRGB=None):
        GraphicsContextBase.set_foreground(self, fg, isRGB)
        self.ctx.set_source_rgb(*self._rgb)


    def set_graylevel(self, frac):
        GraphicsContextBase.set_graylevel(self, frac)
        self.ctx.set_source_rgb(*self._rgb)
        

    def set_joinstyle(self, js):
        if js in ('miter', 'round', 'bevel'):
            self._joinstyle = js
            self.ctx.set_line_join(self._joind[js])
        else:
            raise ValueError('Unrecognized join style.  Found %s' % js)


    def set_linewidth(self, w):
        self._linewidth = w
        self.ctx.set_line_width (self.renderer.points_to_pixels(w))

        
def new_figure_manager(num, *args, **kwargs): # called by backends/__init__.py
    """
    Create a new figure manager instance
    """
    if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
    thisFig = Figure(*args, **kwargs)
    canvas  = FigureCanvasCairo(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


def print_figure_fn(figure, filename, dpi=150, facecolor='w', edgecolor='w',
                    orientation='portrait'):
    if _debug: print _fn_name()

    # settings for printing
    figure.dpi.set(dpi)
    figure.set_facecolor(facecolor)
    figure.set_edgecolor(edgecolor)        

    #if isinstance(filename, file):   # eg when do savefig(sys.stdout)
    #    _save_png (figure, filename) # assume PNG format
    #else:
    if True:
        root, ext = os.path.splitext(filename)       
        ext = ext[1:]
        if ext == '':
            ext      = IMAGE_FORMAT_DEFAULT
            filename = filename + '.' + ext

        ext = ext.lower()
        if ext in ('pdf', 'png', 'ps'):  # native formats
            #try:
            #    fileObject = file(filename,'wb')
            #except IOError, exc:
            #    warnings.warn("%s: %s" % (exc.filename, exc.strerror))
            #else:
            
            #if ext == 'png': _save_png (figure, fileObject)
            if ext == 'png': _save_png (figure, filename)
            #else:            _save_ps_pdf (figure, fileObject, ext,
            #                               orientation)
            else:            _save_ps_pdf (figure, filename, ext, orientation)
            #fileObject.close()
            
        elif ext in ('eps', 'svg'): # backend_svg/ps
            if ext == 'svg':
                from backend_svg import FigureCanvasSVG as FigureCanvas
            else:
                from backend_ps import FigureCanvasPS  as FigureCanvas
            fc = FigureCanvas(figure)
            fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)

        else:
            warnings.warn('Format "%s" is not supported.\nSupported formats: '
                          '%s.' % (ext, ', '.join(IMAGE_FORMAT)))

        
def _save_png (figure, filename):
    width, height = figure.get_width_height()
    width, height = int(width), int(height)

    renderer = RendererCairo (figure.dpi)
    renderer._set_width_height (width, height)
    renderer.surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, width, height)

    figure.draw (renderer)
    renderer.surface.write_to_png (filename)
        

def _save_ps_pdf (figure, filename, ext, orientation):
    # Cairo produces PostScript Level 3
    # 'ggv' can't read cairo ps files, but 'gv' can

    dpi = 72
    figure.dpi.set (dpi)
    w_in, h_in = figure.get_size_inches()
    width_in_points, height_in_points = w_in * dpi, h_in * dpi
    
    if orientation == 'landscape':
        width_in_points, height_in_points = height_in_points, width_in_points
        
    if ext == 'ps':
        surface = cairo.PSSurface (filename, width_in_points, height_in_points)
    else: # pdf
        surface = cairo.PDFSurface (filename, width_in_points,
                                    height_in_points)
    # surface.set_dpi() can be used
    ctx = cairo.Context (surface)
    
    if orientation == 'landscape':
        ctx.rotate (numx.pi/2)
        ctx.translate (0, -height_in_points)
        # cairo/src/cairo_ps_surface.c
        # '%%Orientation: Portrait' is always written to the file header
        # '%%Orientation: Landscape' would possibly cause problems
        # since some printers would rotate again ?
        # TODO:
        # add portrait/landscape checkbox to FileChooser

    renderer = RendererCairo (figure.dpi)
    renderer._set_width_height (width_in_points, height_in_points)
    renderer.surface = ctx.get_target()
    figure.draw (renderer)

    show_fig_border = False  # for testing figure orientation and scaling
    if show_fig_border:
        ctx.new_path()
        ctx.rectangle(0, 0, width_in_points, height_in_points)
        ctx.set_line_width(4.0)
        ctx.set_source_rgb(1,0,0)
        ctx.stroke()
        ctx.move_to(30,30)
        ctx.select_font_face ('sans-serif')
        ctx.set_font_size(20)
        ctx.show_text('Origin corner')
    
    ctx.show_page()


class FigureCanvasCairo(FigureCanvasBase):
    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):
        print_figure_fn(self.figure, filename, dpi, facecolor, edgecolor,
                      orientation)
