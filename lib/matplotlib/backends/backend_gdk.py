from __future__ import division

import os, math
import sys
def fn_name(): return sys._getframe(1).f_code.co_name

import matplotlib
from matplotlib import verbose
from matplotlib.numerix import asarray, fromstring, UInt8, zeros, \
     where, transpose, nonzero, indices, ones, nx

import matplotlib.numerix as numerix
from matplotlib.cbook import is_string_like, enumerate, onetrue
from matplotlib.font_manager import fontManager

from matplotlib.backend_bases import RendererBase, GraphicsContextBase, \
     FigureManagerBase, FigureCanvasBase, error_msg
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from matplotlib.mathtext import math_parse_s_ft2font

pygtk_version_required = (1,99,16)
try:
    import pygtk
    if not matplotlib.FROZEN:
        pygtk.require('2.0')
except:
    print >> sys.stderr, sys.exc_info()[1]
    raise SystemExit('PyGTK version %d.%d.%d or greater is required to run the GTK Matplotlib backends'
                     % pygtk_version_required)

import gtk, gobject, pango
from gtk import gdk
if gtk.pygtk_version < pygtk_version_required:
    raise SystemExit ("PyGTK %d.%d.%d is installed\n"
                      "PyGTK %d.%d.%d or later is required"
                      % (gtk.pygtk_version + pygtk_version_required))
backend_version = "%d.%d.%d" % gtk.pygtk_version
del pygtk_version_required


DEBUG = False

# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 96

# Image formats that this backend supports - for FileChooser and print_figure()
IMAGE_FORMAT          = ['eps', 'jpg', 'png', 'ps', 'svg'] + ['bmp'] # , 'raw', 'rgb']
IMAGE_FORMAT.sort()
IMAGE_FORMAT_DEFAULT  = 'png'


class RendererGDK(RendererBase):
    fontweights = {
        100          : pango.WEIGHT_ULTRALIGHT,
        200          : pango.WEIGHT_LIGHT,
        300          : pango.WEIGHT_LIGHT,
        400          : pango.WEIGHT_NORMAL,
        500          : pango.WEIGHT_NORMAL,
        600          : pango.WEIGHT_BOLD,
        700          : pango.WEIGHT_BOLD,
        800          : pango.WEIGHT_HEAVY,
        900          : pango.WEIGHT_ULTRABOLD,
        'ultralight' : pango.WEIGHT_ULTRALIGHT,
        'light'      : pango.WEIGHT_LIGHT,
        'normal'     : pango.WEIGHT_NORMAL,
        'medium'     : pango.WEIGHT_NORMAL,
        'semibold'   : pango.WEIGHT_BOLD,
        'bold'       : pango.WEIGHT_BOLD,
        'heavy'      : pango.WEIGHT_HEAVY,
        'ultrabold'  : pango.WEIGHT_ULTRABOLD,
        'black'      : pango.WEIGHT_ULTRABOLD,
                   }

    # cache for efficiency, these must be at class, not instance level
    layoutd = {}  # a map from text prop tups to pango layouts
    rotated = {}  # a map from text prop tups to rotated text pixbufs

    def __init__(self, gtkDA, dpi):
        # gtkDA is used in '<widget>.create_pango_layout(s)' only
        self.gtkDA = gtkDA
        self.dpi   = dpi
        self._cmap = gtkDA.get_colormap()
        
    def _set_pixmap (self, gdkDrawable):
        self.gdkDrawable = gdkDrawable

    def _set_width_height (self, width, height):
        """w,h is the figure w,h not the pixmap w,h
        """
        self.width, self.height = width, height

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
        x, y = int(x-0.5*width), self.height-int(y+0.5*height)
        w, h = int(width)+1, int(height)+1
        a1, a2 = int(angle1*64), int(angle2*64)
        
        if rgbFace:
            saveColor = gc.gdkGC.foreground
            gc.gdkGC.foreground = gc.rgb_to_gdk_color(rgbFace)
            self.gdkDrawable.draw_arc(gc.gdkGC, True, x, y, w, h, a1, a2)
            gc.gdkGC.foreground = saveColor
        self.gdkDrawable.draw_arc(gc.gdkGC, False, x, y, w, h, a1, a2)


    def draw_image(self, x, y, im, origin, bbox):
        if bbox != None:
            l,b,w,h = bbox.get_bounds()
            #rectangle = (int(l), self.height-int(b+h),
            #             int(w), int(h))
            # set clip rect?

        flipud = origin=='lower'
        rows, cols, s = im.as_str(flipud)

        X = fromstring(s, UInt8)
        X.shape = rows, cols, 4

        pb=gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,
                          has_alpha=1, bits_per_sample=8,
                          width=cols, height=rows)
        try:
            pa = pb.get_pixels_array()
        except AttributeError:
            pa = pb.pixel_array
        except RuntimeError, exc: #  pygtk was not compiled with Numeric Python support
            verbose.report_error('Error: %s' % exc)
            return

        pa[:,:,:] = X

        gc = self.new_gc()

        if flipud:  y = self.height-y-rows

        pb.render_to_drawable(self.gdkDrawable, gc.gdkGC, 0, 0,
                              int(x), int(y), cols, rows,
                              gdk.RGB_DITHER_NONE, 0, 0)

            
    def draw_line(self, gc, x1, y1, x2, y2):
        self.gdkDrawable.draw_line(gc.gdkGC, int(x1), self.height-int(y1),
                                   int(x2), self.height-int(y2))


    def draw_lines(self, gc, x, y):
        x = x.astype(nx.Int16)
        #y = self.height*ones(y.shape, nx.Int16) - y.astype(nx.Int16)  
        y = self.height - y.astype(nx.Int16)  
        self.gdkDrawable.draw_lines(gc.gdkGC, zip(x,y))
        

    def draw_point(self, gc, x, y):
        self.gdkDrawable.draw_point(gc.gdkGC, int(x), self.height-int(y))


    def draw_polygon(self, gc, rgbFace, points):
        points = [(int(x), self.height-int(y)) for x,y in points]
        if rgbFace:
            saveColor = gc.gdkGC.foreground
            gc.gdkGC.foreground = gc.rgb_to_gdk_color(rgbFace)
            self.gdkDrawable.draw_polygon(gc.gdkGC, True, points)
            gc.gdkGC.foreground = saveColor
        self.gdkDrawable.draw_polygon(gc.gdkGC, False, points)


    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        x, y = int(x), self.height-int(y+height)
        #x, y = int(x), self.height-int(math.ceil(y+height))
        w, h = int(math.ceil(width)), int(math.ceil(height))

        if rgbFace:
            saveColor = gc.gdkGC.foreground
            gc.gdkGC.foreground = gc.rgb_to_gdk_color(rgbFace)
            self.gdkDrawable.draw_rectangle(gc.gdkGC, True, x, y, w, h)
            gc.gdkGC.foreground = saveColor
        self.gdkDrawable.draw_rectangle(gc.gdkGC, False, x, y, w, h)


    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        x, y = int(x), int(y)

        if angle not in (0,90):
            verbose.report_error('The GTK backend cannot draw text at a %i degree angle, try GtkAgg instead' % angle)

        elif ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)

        elif angle==90:
            self._draw_rotated_text(gc, x, y, s, prop, angle)

        else:
            layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
            l, b, w, h = inkRect
            self.gdkDrawable.draw_layout(gc.gdkGC, x, y-h-b, layout)

        
    def _draw_mathtext(self, gc, x, y, s, prop, angle):
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
        Xs = numerix.max(Xall,1)

        # convert it to it's proper shape
        Xs.shape = imh, imw

        pb=gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,
                          has_alpha=1, bits_per_sample=8, width=imw, height=imh)

        try:
            pa = pb.get_pixels_array()
        except AttributeError:
            pa = pb.pixel_array
        except RuntimeError, exc: #  'pygtk was not compiled with Numeric Python support'
            verbose.report_error('mathtext not supported: %s' % exc)
            return        

        rgb = gc.get_rgb()
        pa[:,:,0]=int(rgb[0]*255)
        pa[:,:,1]=int(rgb[1]*255)
        pa[:,:,2]=int(rgb[2]*255)
        pa[:,:,3]=Xs

        pb.render_to_drawable(self.gdkDrawable, gc.gdkGC, 0, 0,
                              int(x), int(y), imw, imh,
                              gdk.RGB_DITHER_NONE, 0, 0)
            
        
    def _draw_rotated_text(self, gc, x, y, s, prop, angle):
        """
        Draw the text rotated 90 degrees, other angles are not supported
        """
        # this function (and its called functions) is a bottleneck
        # Pango 1.6 supports rotated text, but pygtk 2.4.0 does not yet have wrapper functions
        # GTK+ 2.6 pixbufs support rotation

        gdrawable = self.gdkDrawable
        ggc = gc.gdkGC

        layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
        l, b, w, h = inkRect
        x = int(x-h)
        y = int(y-w)

        key = (x,y,s,angle,hash(prop))
        imageVert = self.rotated.get(key)
        if imageVert != None:
            gdrawable.draw_image(ggc, imageVert, 0, 0, x, y, h, w)
            return

        imageBack = gdrawable.get_image(x, y, w, h)
        imageVert = gdrawable.get_image(x, y, h, w)
        #imageFlip = gtk.gdk.Image(type=gdk.IMAGE_NORMAL,
        imageFlip = gtk.gdk.Image(type=gdk.IMAGE_FASTEST,
                                  visual=gdrawable.get_visual(),
                                  width=w, height=h)
        if imageFlip == None or imageBack == None or imageVert == None:
            verbose.report_error("Could not renderer vertical text", s)
            return
        imageFlip.set_colormap(self._cmap)
        for i in range(w):
            for j in range(h):
                imageFlip.put_pixel(i, j, imageVert.get_pixel(j,w-i-1) )

        gdrawable.draw_image(ggc, imageFlip, 0, 0, x, y, w, h)
        gdrawable.draw_layout(ggc, x, y-b, layout)

        imageIn  = gdrawable.get_image(x, y, w, h)
        for i in range(w):
            for j in range(h):
                imageVert.put_pixel(j, i, imageIn.get_pixel(w-i-1,j) )

        gdrawable.draw_image(ggc, imageBack, 0, 0, x, y, w, h)
        gdrawable.draw_image(ggc, imageVert, 0, 0, x, y, h, w)
        self.rotated[key] = imageVert


    def _get_pango_layout(self, s, prop):
        """
        Create a pango layout instance for Text 's' with properties 'prop'.
        Return - pango layout (from cache if already exists)
        """
        # problem? - cache gets bigger and bigger, is never cleared out
        # two (not one) layouts are created for every text item s (then they are cached) - why?
        
        key = self.dpi.get(), s, hash(prop)
        value = self.layoutd.get(key)
        if value != None:
            #print 'layoutd cached value'
            return value
        #print 'len(self.layoutd), s', len(self.layoutd), s

        size = prop.get_size_in_points() * self.dpi.get() / PIXELS_PER_INCH
        size = round(size)

        font_str = '%s, %s %i' % (prop.get_name(), prop.get_style(), size,)
        font = pango.FontDescription(font_str)

        # later - add fontweight to font_str
        font.set_weight(self.fontweights[prop.get_weight()])

        layout = self.gtkDA.create_pango_layout(s)
        layout.set_font_description(font)    
        inkRect, logicalRect = layout.get_pixel_extents()
        
        self.layoutd[key] = layout, inkRect, logicalRect
        return layout, inkRect, logicalRect


    def flipy(self):
        return True

    def get_canvas_width_height(self):
        #return self.gtkDA.allocation.width, self.gtkDA.allocation.height        
        return self.width, self.height

    def get_text_width_height(self, s, prop, ismath):
        if ismath:
            width, height, fonts = math_parse_s_ft2font(
                s, self.dpi.get(), prop.get_size_in_points())
            return width, height

        layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
        l, b, w, h = inkRect
        return w, h+1

    def new_gc(self):
        return GraphicsContextGDK(renderer=self)


    def points_to_pixels(self, points):
        return points * PIXELS_PER_INCH/72.0 * self.dpi.get()/72.0


class GraphicsContextGDK(GraphicsContextBase):
    # a cache shared by all class instances
    _cached = {}  # map: rgb color -> gdk.Color

    _joind = {
        'bevel' : gdk.JOIN_BEVEL,
        'miter' : gdk.JOIN_MITER,
        'round' : gdk.JOIN_ROUND,
        }

    _capd = {
        'butt'       : gdk.CAP_BUTT,
        'projecting' : gdk.CAP_PROJECTING,
        'round'      : gdk.CAP_ROUND,
        }

              
    def __init__(self, renderer):
        GraphicsContextBase.__init__(self)
        self.renderer = renderer
        self.gdkGC    = gtk.gdk.GC(renderer.gdkDrawable)
        self._cmap    = renderer._cmap


    def rgb_to_gdk_color(self, rgb):
        """
        rgb - an RGB tuple (three 0.0-1.0 values)
        return an allocated gtk.gdk.Color
        """
        try:
            return self._cached[rgb] 
        except KeyError:
            color = self._cached[rgb] = \
                    self._cmap.alloc_color(
                        int(rgb[0]*65535),int(rgb[1]*65535),int(rgb[2]*65535))
            return color


    def set_capstyle(self, cs):
        GraphicsContextBase.set_capstyle(self, cs)
        self.gdkGC.cap_style = self._capd[self._capstyle]


    def set_clip_rectangle(self, rectangle):
        GraphicsContextBase.set_clip_rectangle(self, rectangle)
        l,b,w,h = rectangle
        rectangle = (int(l), self.renderer.height-int(b+h)+1,
                     int(w), int(h))
        #rectangle = (int(l), self.renderer.height-int(b+h),
        #             int(w+1), int(h+2))
        self.gdkGC.set_clip_rectangle(rectangle)        


    def set_dashes(self, dash_offset, dash_list):
        GraphicsContextBase.set_dashes(self, dash_offset, dash_list)

        if dash_list == None:
            self.gdkGC.line_style = gdk.LINE_SOLID
        else:
            pixels = self.renderer.points_to_pixels(asarray(dash_list))
            dl = [max(1, int(round(val))) for val in pixels]
            self.gdkGC.set_dashes(dash_offset, dl)
            self.gdkGC.line_style = gdk.LINE_ON_OFF_DASH


    def set_foreground(self, fg, isRGB=False):
        GraphicsContextBase.set_foreground(self, fg, isRGB)
        self.gdkGC.foreground = self.rgb_to_gdk_color(self.get_rgb())


    def set_graylevel(self, frac):
        GraphicsContextBase.set_graylevel(self, frac)
        self.gdkGC.foreground = self.rgb_to_gdk_color(self.get_rgb())
        

    def set_joinstyle(self, js):
        GraphicsContextBase.set_joinstyle(self, js)
        self.gdkGC.join_style = self._joind[self._joinstyle]


    def set_linewidth(self, w):
        GraphicsContextBase.set_linewidth(self, w)
        pixels = self.renderer.points_to_pixels(w)
        self.gdkGC.line_width = max(1, int(round(pixels)))

                                               
def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    thisFig = Figure(*args, **kwargs)
    canvas  = FigureCanvasGDK(thisFig)
    manager = FigureManagerBase(canvas, num)
    # equals:
    #manager = FigureManagerBase (FigureCanvasGDK (Figure(*args, **kwargs), num)
    return manager


class FigureCanvasGDK(FigureCanvasBase):
    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)
        self._pixmap_width  = -1
        self._pixmap_height = -1
        
        self._renderer_init()

    def _renderer_init(self):
        #self._renderer = RendererGDK (self, self.figure.dpi) # self is no longer a widget subclass
        self._renderer = RendererGDK (gtk.DrawingArea(), self.figure.dpi)

        
    def _render_to_pixmap(self, width, height):
        """Render the figure to a gdk.Pixmap, is used for
           - rendering the pixmap to display        (pylab.draw)
           - rendering the pixmap to save to a file (pylab.savefig)
        Should not be overridden
        """
        if DEBUG: print 'FigureCanvasGDK.%s' % fn_name()
        create_pixmap = False
        if width > self._pixmap_width:
            # increase the pixmap in 10%+ (rather than 1 pixel) steps
            self._pixmap_width  = max (int (self._pixmap_width  * 1.1), width)
            create_pixmap = True

        if height > self._pixmap_height:
            self._pixmap_height = max (int (self._pixmap_height * 1.1), height)
            create_pixmap = True

        if create_pixmap:
            if DEBUG: print 'FigureCanvasGTK.%s new pixmap' % fn_name()
            #self._pixmap = gtk.gdk.Pixmap (self.window, self._pixmap_width,
            self._pixmap = gtk.gdk.Pixmap (None, self._pixmap_width,
                                           self._pixmap_height, depth=24)
            # gtk backend must use self.window
            self._renderer._set_pixmap (self._pixmap)

        self._renderer._set_width_height (width, height)
        self.figure.draw (self._renderer)


    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):
        root, ext = os.path.splitext(filename)       
        ext = ext[1:]
        if ext == '':
            ext      = IMAGE_FORMAT_DEFAULT
            filename = filename + '.' + ext        

        self.figure.dpi.set(dpi)        
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        ext = ext.lower()
        if ext in ('jpg', 'png'):          # native printing
            width, height = self.figure.get_width_height()
            width, height = int(width), int(height)
            self._render_to_pixmap(width, height)

            # jpg colors don't match the display very well, png colors match better
            pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, 0, 8,
                                    width, height)
            #pixbuf.get_from_drawable(self._pixmap, self._pixmap.get_colormap(),
            pixbuf.get_from_drawable(self._pixmap, self._renderer._cmap,
                                     0, 0, 0, 0, width, height)
        
            # pixbuf.save() recognises 'jpeg' not 'jpg'
            if ext == 'jpg': ext = 'jpeg' 
            try:
                pixbuf.save(filename, ext)
            except gobject.GError, exc:
                error_msg('Save figure failure:\n%s' % (exc,), parent=self)

        elif ext in ('eps', 'ps', 'svg',):
            if ext == 'svg':
                from backend_svg import FigureCanvasSVG as FigureCanvas
            else:
                from backend_ps  import FigureCanvasPS  as FigureCanvas

            try:
                fc = self.switch_backends(FigureCanvas)
                fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)
            except IOError, exc:
                error_msg("Save figure failure:\n%s: %s" %
                          (exc.filename, exc.strerror), parent=self)
            except Exception, exc:
                error_msg("Save figure failure:\n%s" % exc, parent=self)

        elif ext in ('bmp', 'raw', 'rgb',):
            try: 
                from backend_agg import FigureCanvasAgg  as FigureCanvas
            except:
                error_msg('Save figure failure:\n'
                          'Agg must be loaded to save as bmp, raw and rgb',
                          parent=self)                
            else:
                fc = self.switch_backends(FigureCanvas)
                fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)

        else:
            error_msg('Format "%s" is not supported.\nSupported formats are %s.' %
                      (ext, ', '.join(IMAGE_FORMAT)),
                      parent=self)
