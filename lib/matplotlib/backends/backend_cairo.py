"""
A Cairo backend for matplotlib
Author: Steve Chaplin
 
Cairo is a vector graphics library with cross-device output support.
Features of Cairo:
 * anti-aliasing
 * alpha channel 
 * in-memory image buffers
 * save image files:
   - PNG, PostScript, PDF

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
import sys, warnings
def _fn_name(): return sys._getframe(1).f_code.co_name

from matplotlib.numerix import asarray, pi, fromstring, UInt8, zeros
import matplotlib.numerix as numerix
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook      import enumerate
from matplotlib.figure     import Figure
from matplotlib.mathtext   import math_parse_s_ft2font
from matplotlib.transforms import Bbox

import cairo

version_required = (0,1,4)
try: cairo.version_info
except AttributeError:  # remove at version 0.1.5 or 6
   backend_version = 'Unknown'
else:
   if cairo.version_info < version_required:
      raise SystemExit ("PyCairo %d.%d.%d is installed\n"
                        "PyCairo %d.%d.%d or later is required"
                        % (cairo.version_info + version_required))
   backend_version = cairo.version
   del version_required

HAVE_CAIRO_NUMPY = True
try:
   import cairo.numpy
except:
   HAVE_CAIRO_NUMPY = False


DEBUG = False

# the true dots per inch on the screen; should be display dependent
PIXELS_PER_INCH = 96

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
    

    def __init__(self, matrix, dpi):
        """width, height - the canvas width, height. Is not necessarily
        the same as the surface (pixmap) width, height
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        self.matrix   = matrix
        self.dpi      = dpi
        self.text_ctx = cairo.Context()

    def _set_pixmap(self, pixmap):
        # note: this is a cairo.gtk function, its called by backend_gtkcairo.py
        self.surface = cairo.gtk.surface_create_for_drawable (pixmap)

    def _set_width_height(self, width, height):
        self.width    = width
        self.height   = height

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
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
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()

        if numerix.which[0] == "numarray":
            warnings.warn("draw_image() currently works for numpy, but not numarray")
            return

        if not HAVE_CAIRO_NUMPY:
            warnings.warn("cairo.numpy module required for draw_image()")
            return

        # bbox - not currently used
        flipud = origin=='lower'  # not currently used

        ctx = cairo.Context()
        ctx.set_target_surface (self.surface)
        ctx.set_matrix (self.matrix)

        rows, cols, buf = im.buffer_argb32()  # ARGB32, but colors still wrong
        X = fromstring(buf, UInt8)
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

        # works for numpy X, not numarray X
        surface = cairo.numpy.surface_create_for_array (X)

        # Alternative
        #surface = cairo.surface_create_for_image(buf, cairo.FORMAT_ARGB32, cols, rows) #, stride)
        # error: TypeError: Cannot use string as modifiable buffer

        ctx.translate (x,y)
        ctx.show_surface (surface, cols, rows)


    def draw_line(self, gc, x1, y1, x2, y2):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        ctx = gc.ctx
        ctx.new_path()
        ctx.move_to (x1, self.height - y1)
        ctx.line_to (x2, self.height - y2)
        ctx.stroke()


    #def draw_lines(self, gc, x, y):
    def draw_lines(self, gc, x, y, transform=None):
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


    #def draw_markers(self, gc, path, x, y, transform):
    def _draw_markers(self, gc, path, x, y, transform): # disable, not finished yet
        """
        Draw the markers defined by path at each of the positions in x
        and y.  path coordinates are points, x and y coords will be
        transformed by the transform
        """
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        from matplotlib.path import STOP, MOVETO, LINETO, CURVE3, CURVE4, ENDPOLY

        if transform.need_nonlinear():
            x,y = transform.nonlinear_only_numerix(x, y)

        # the a,b,c,d,tx,ty affine which transforms x and y
        vec6 = transform.as_vec6_val()
        # this defines a single vertex.  We need to define this as ps
        # function, properly stroked and filled with linewidth etc,
        # and then simply iterate over the x and y and call this
        # function at each position.  Eg, this is the path that is
        # relative to each x and y offset.

        ctx = gc.ctx

        def draw_path():
            # could trace path just once, then save/restore() ?
            for p in path:
               code = p[0]
               if code==MOVETO:
                  mx, my = p[1:]
                  # ctx.move_to (mx, self.height - my)
                  ctx.move_to (mx, my)
               elif code==LINETO:
                  mx, my = p[1:]
                  # ctx.line_to (mx, self.height - my)                
                  ctx.line_to (mx, my)                
               elif code==ENDPOLY:
                  ctx.close_path()
                  # draw_marker
                  fill = p[1]
                  if fill:  # we can get the fill color here
                     #rgba = p[2:]
                     rgb = p[2:5] # later - set alpha
                     ctx.save()
                     ctx.set_rgb_color (*rgb)
                     ctx.fill()
                     ctx.restore()
            
                  ctx.stroke() # later stroke and/or fill, sel fill color
                  break
                    
        # the gc contains the stroke width and color as always
        for i in xrange(len(x)):
            # FIXME
            # 1) markers are upside down
            # 2) get line-dash working
            # -> cmp to plot w/o draw_markers()
            # need to translate points to pixels?
            # look to writing in a more efficient way
            # use Cairo transform
            
            ctx.save()
            thisx, thisy = x[i], y[i]
            
            # apply affine transform x and y to define marker center
            ctx.new_path()
            tx, ty = transform.xy_tup((thisx, thisy))
            ctx.translate(tx, self.height - ty)
            draw_path()

            ctx.restore() # wrap translate()

        
    def draw_point(self, gc, x, y):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # render by drawing a 0.5 radius circle
        gc.ctx.new_path()
        gc.ctx.arc (x, self.height - y, 0.5, 0, 2*pi)
        gc.ctx.fill()


    def draw_polygon(self, gc, rgbFace, points):
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
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()

        if ismath:
           self._draw_mathtext(gc, x, y, s, prop, angle)

        else:
           ctx = gc.ctx
           ctx.new_path()
           ctx.move_to (x, y)
           ctx.select_font (prop.get_name(),
                            self.fontangles [prop.get_style()],
                            self.fontweights[prop.get_weight()])
           # 1.4 scales font to a similar size to GTK / GTKAgg backends
           size = prop.get_size_in_points() * self.dpi.get() / PIXELS_PER_INCH * 1.4

           ctx.save()
           if angle:
              ctx.rotate (-angle * pi / 180)
           ctx.scale_font (size)
           ctx.show_text (s)
           ctx.restore()


    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
       # mathtext using the gtk/gdk method

        if numerix.which[0] == "numarray":
            warnings.warn("_draw_mathtext() currently works for numpy, but not numarray")
            return

        if not HAVE_CAIRO_NUMPY:
            warnings.warn("cairo.numpy module required for _draw_mathtext()")
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
        Xall = zeros((N,len(fonts)), typecode=UInt8)

        for i, font in enumerate(fonts):
            if angle == 90:
                font.horiz_image_to_vert_image() # <-- Rotate
            imw, imh, s = font.image_as_str()
            Xall[:,i] = fromstring(s, UInt8)  

        # get the max alpha at each pixel
        Xs = numerix.mlab.max(Xall,1)

        # convert it to it's proper shape
        Xs.shape = imh, imw

        pa = zeros(shape=(imh,imw,4), typecode=UInt8)
        rgb = gc.get_rgb()
        pa[:,:,0] = int(rgb[0]*255)
        pa[:,:,1] = int(rgb[1]*255)
        pa[:,:,2] = int(rgb[2]*255)
        pa[:,:,3] = Xs

        # works for numpy pa, not a numarray pa
        surface = cairo.numpy.surface_create_for_array(pa)
        gc.ctx.translate (x,y)
        gc.ctx.show_surface (surface, imw, imh)
        # should really restore state before translate?
            

    def flipy(self):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        return True

    
    def get_canvas_width_height(self):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        return self.width, self.height
    

    def get_text_width_height(self, s, prop, ismath):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        if ismath:
            width, height, fonts = math_parse_s_ft2font(
               s, self.dpi.get(), prop.get_size_in_points())
            return width, height

        ctx = self.text_ctx
        ctx.save()
        ctx.select_font (prop.get_name(),
                         self.fontangles [prop.get_style()],
                         self.fontweights[prop.get_weight()])

        # 1.4 scales font to a similar size to GTK / GTKAgg backends
        size = prop.get_size_in_points() * self.dpi.get() / PIXELS_PER_INCH * 1.4
        # problem - scale remembers last setting and font can become
        # enormous causing program to crash
        # save/restore prevents the problem
        ctx.scale_font (size)
        
        w, h = ctx.text_extents (s)[2:4]
        ctx.restore()
            
        return w, h

                              
    def new_gc(self):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        gc = GraphicsContextCairo (renderer=self)
        gc.ctx.set_target_surface (self.surface)
        gc.ctx.set_matrix (self.matrix)
        return gc


    def points_to_pixels(self, points):
        if DEBUG: print 'backend_cairo.RendererCairo.%s()' % _fn_name()
        # is this correct for cairo? (copied from gtk)
        return points * PIXELS_PER_INCH/72.0 * self.dpi.get()/72.0


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

    
    def __init__(self, renderer):
        GraphicsContextBase.__init__(self)
        self.renderer = renderer
        self.ctx = cairo.Context()

        
    def set_alpha(self, alpha):
        self._alpha = alpha
        self.ctx.set_alpha(alpha)


    def set_capstyle(self, cs):
        if cs in ('butt', 'round', 'projecting'):
            self._capstyle = cs
            self.ctx.set_line_cap (self._capd[cs])
        else:
            raise ValueError('Unrecognized cap style.  Found %s' % cs)


    def set_clip_rectangle(self, rectangle):
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

        #ctx.clip ()
        

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes == None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            dashes_pixels = self.renderer.points_to_pixels(asarray(dashes))
            self.ctx.set_dash(dashes_pixels, offset)
        

    def set_foreground(self, fg, isRGB=None):
        GraphicsContextBase.set_foreground(self, fg, isRGB)
        self.ctx.set_rgb_color(*self._rgb)


    def set_graylevel(self, frac):
        GraphicsContextBase.set_graylevel(self, frac)
        self.ctx.set_rgb_color(*self._rgb)
        

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
    if DEBUG: print 'backend_cairo.%s()' % _fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas  = FigureCanvasCairo(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


def print_figure_fn(figure, filename, dpi=150, facecolor='w', edgecolor='w',
                    orientation='portrait'):
    if DEBUG: print 'backend_cairo.FigureCanvasCairo.%s()' % _fn_name()

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
        if ext in ('pdf', 'png', 'ps'):  # native formats
            try:
                fileObject = file(filename,'wb')
            except IOError, exc:
                warnings.warn("%s: %s" % (exc.filename, exc.strerror))
            else:
                if ext == 'png': _save_png (figure, fileObject)
                else:            _save_ps_pdf (figure, fileObject, ext, orientation)
                fileObject.close()
            
        elif ext in ('eps', 'svg'): # backend_svg/ps
            if ext == 'svg':
                from backend_svg import FigureCanvasSVG as FigureCanvas
            else:
                from backend_ps import FigureCanvasPS  as FigureCanvas
            fc = FigureCanvas(figure)
            fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)

        else:
            warnings.warn('Format "%s" is not supported.\nSupported formats: %s.' %
                          (ext, ', '.join(IMAGE_FORMAT)))

        
def _save_png (figure, fileObject):
    width, height = figure.get_width_height()
    width, height = int(width), int(height)

    ctx = cairo.Context()
    ctx.set_target_png (fileObject, cairo.FORMAT_ARGB32, width, height)

    renderer = RendererCairo (ctx.matrix, figure.dpi)
    renderer._set_width_height(width, height)
    renderer.surface = ctx.target_surface
    figure.draw(renderer)
    ctx.show_page()
        

def _save_ps_pdf (figure, fileObject, ext, orientation):
    # Cairo produces PostScript Level 3
    # 'ggv' can't read cairo ps files, but 'gv' can

    ppi = 200.0
    #figure.dpi.set(72)
    figure.dpi.set(96) # Cairo uses 96 dpi

    w_in, h_in = figure.get_size_inches()
    width, height = figure.get_width_height()
    
    ctx = cairo.Context()

    if orientation == 'landscape':
        w_in, h_in = h_in, w_in
        
    if ext == 'ps':
        ctx.set_target_ps (fileObject, w_in, h_in, ppi, ppi)
    else: # pdf
        ctx.set_target_pdf (fileObject, w_in, h_in, ppi, ppi)

    if orientation == 'landscape':
        ctx.rotate(pi/2)
        ctx.translate(0, -height)
        # cairo/src/cairo_ps_surface.c
        # '%%Orientation: Portrait' is always written to the file header
        # '%%Orientation: Landscape' would possibly cause problems
        # since some printers would rotate again ?
        # TODO:
        # add portrait/landscape checkbox to FileChooser

    renderer = RendererCairo (ctx.matrix, figure.dpi)
    renderer._set_width_height(width, height)
    renderer.surface = ctx.target_surface
    figure.draw(renderer)

    show_fig_border = False  # for testing figure orientation and scaling
    if show_fig_border:
        ctx.new_path()
        ctx.rectangle(0, 0, width, height)
        ctx.set_line_width(4.0)
        ctx.set_rgb_color(1,0,0)
        ctx.stroke()
        ctx.move_to(30,30)
        ctx.select_font('sans-serif')
        ctx.scale_font(20)
        ctx.show_text('Origin corner')
    
    ctx.show_page()


class FigureCanvasCairo(FigureCanvasBase):
    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):
        print_figure_fn(self.figure, filename, dpi, facecolor, edgecolor,
                      orientation)
