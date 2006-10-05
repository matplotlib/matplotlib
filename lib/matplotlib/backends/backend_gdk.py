from __future__ import division

import math
import os
import sys
import warnings
def fn_name(): return sys._getframe(1).f_code.co_name

import gobject
import gtk; gdk = gtk.gdk
import pango
pygtk_version_required = (2,2,0)
if gtk.pygtk_version < pygtk_version_required:
    raise SystemExit ("PyGTK %d.%d.%d is installed\n"
                      "PyGTK %d.%d.%d or later is required"
                      % (gtk.pygtk_version + pygtk_version_required))
del pygtk_version_required

import matplotlib
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase, \
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import is_string_like, enumerate
from matplotlib.figure import Figure
from matplotlib.mathtext import math_parse_s_ft2font
import matplotlib.numerix as numerix
from matplotlib.numerix import asarray, fromstring, UInt8, zeros, \
     where, transpose, nonzero, indices, ones, nx

if numerix.which[0] == "numarray":
    from matplotlib.backends._na_backend_gdk import pixbuf_get_pixels_array
elif numerix.which[0] == "numeric":
    from matplotlib.backends._nc_backend_gdk import pixbuf_get_pixels_array
else:
    from matplotlib.backends._ns_backend_gdk import pixbuf_get_pixels_array


backend_version = "%d.%d.%d" % gtk.pygtk_version
_debug = False

# Image formats that this backend supports - for FileChooser and print_figure()
IMAGE_FORMAT  = ['eps', 'jpg', 'png', 'ps', 'svg'] + ['bmp'] # , 'raw', 'rgb']
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
        # widget gtkDA is used for:
        #  '<widget>.create_pango_layout(s)'
        #  cmap line below)
        self.gtkDA = gtkDA
        self.dpi   = dpi
        self._cmap = gtkDA.get_colormap()

    def set_pixmap (self, pixmap):
        self.gdkDrawable = pixmap

    def set_width_height (self, width, height):
        """w,h is the figure w,h not the pixmap w,h
        """
        self.width, self.height = width, height

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2, rotation):
        x, y = int(x-0.5*width), self.height-int(y+0.5*height)
        w, h = int(width)+1, int(height)+1
        a1, a2 = int(angle1*64), int(angle2*64)

        if rgbFace:
            saveColor = gc.gdkGC.foreground
            gc.gdkGC.foreground = gc.rgb_to_gdk_color(rgbFace)
            self.gdkDrawable.draw_arc(gc.gdkGC, True, x, y, w, h, a1, a2)
            gc.gdkGC.foreground = saveColor
        if gc.gdkGC.line_width > 0:
            self.gdkDrawable.draw_arc(gc.gdkGC, False, x, y, w, h, a1, a2)


    def draw_image(self, x, y, im, bbox):
        if bbox != None:
            l,b,w,h = bbox.get_bounds()
            #rectangle = (int(l), self.height-int(b+h),
            #             int(w), int(h))
            # set clip rect?

        im.flipud_out()
        rows, cols, image_str = im.as_rgba_str()

        image_array = fromstring(image_str, UInt8)
        image_array.shape = rows, cols, 4

        pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,
                                has_alpha=True, bits_per_sample=8,
                                width=cols, height=rows)

        array = pixbuf_get_pixels_array(pixbuf)
        array[:,:,:] = image_array

        gc = self.new_gc()


        y = self.height-y-rows

        try: # new in 2.2
            # can use None instead of gc.gdkGC, if don't need clipping
            self.gdkDrawable.draw_pixbuf (gc.gdkGC, pixbuf, 0, 0,
                                          int(x), int(y), cols, rows,
                                          gdk.RGB_DITHER_NONE, 0, 0)
        except AttributeError:
            # deprecated in 2.2
            pixbuf.render_to_drawable(self.gdkDrawable, gc.gdkGC, 0, 0,
                                  int(x), int(y), cols, rows,
                                  gdk.RGB_DITHER_NONE, 0, 0)

        # unflip
        im.flipud_out()


    def draw_line(self, gc, x1, y1, x2, y2):
        if gc.gdkGC.line_width > 0:
            self.gdkDrawable.draw_line(gc.gdkGC, int(x1), self.height-int(y1),
                                   int(x2), self.height-int(y2))


    def draw_lines(self, gc, x, y, transform=None):
        if gc.gdkGC.line_width > 0:
            x = x.astype(nx.Int16)
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
        if gc.gdkGC.line_width > 0:
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
        if gc.gdkGC.line_width > 0:
            self.gdkDrawable.draw_rectangle(gc.gdkGC, False, x, y, w, h)


    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        x, y = int(x), int(y)

        if x <0 or y <0: # window has shrunk and text is off the edge
            return

        if angle not in (0,90):
            warnings.warn('backend_gdk: unable to draw text at angles ' +
                          'other than 0 or 90')
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

        imw, imh, image_str = fonts[0].image_as_str()
        N = imw*imh

        # a numpixels by num fonts array
        Xall = zeros((N,len(fonts)), typecode=UInt8)

        for i, font in enumerate(fonts):
            if angle == 90:
                font.horiz_image_to_vert_image() # <-- Rotate
            imw, imh, image_str = font.image_as_str()
            Xall[:,i] = fromstring(image_str, UInt8)

        # get the max alpha at each pixel
        Xs = numerix.mlab.max(Xall,1)

        # convert it to it's proper shape
        Xs.shape = imh, imw

        pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, has_alpha=True,
                                bits_per_sample=8, width=imw, height=imh)

        array = pixbuf_get_pixels_array(pixbuf)

        rgb = gc.get_rgb()
        array[:,:,0]=int(rgb[0]*255)
        array[:,:,1]=int(rgb[1]*255)
        array[:,:,2]=int(rgb[2]*255)
        array[:,:,3]=Xs

        try: # new in 2.2
            # can use None instead of gc.gdkGC, if don't need clipping
            self.gdkDrawable.draw_pixbuf (gc.gdkGC, pixbuf, 0, 0,
                                          int(x), int(y), imw, imh,
                                          gdk.RGB_DITHER_NONE, 0, 0)
        except AttributeError:
            # deprecated in 2.2
            pixbuf.render_to_drawable(self.gdkDrawable, gc.gdkGC, 0, 0,
                                  int(x), int(y), imw, imh,
                                  gdk.RGB_DITHER_NONE, 0, 0)


    def _draw_rotated_text(self, gc, x, y, s, prop, angle):
        """
        Draw the text rotated 90 degrees, other angles are not supported
        """
        # this function (and its called functions) is a bottleneck
        # Pango 1.6 supports rotated text, but pygtk 2.4.0 does not yet have
        # wrapper functions
        # GTK+ 2.6 pixbufs support rotation

        gdrawable = self.gdkDrawable
        ggc = gc.gdkGC

        layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
        l, b, w, h = inkRect
        x = int(x-h)
        y = int(y-w)

        if x < 0 or y < 0: # window has shrunk and text is off the edge
            return

        key = (x,y,s,angle,hash(prop))
        imageVert = self.rotated.get(key)
        if imageVert != None:
            gdrawable.draw_image(ggc, imageVert, 0, 0, x, y, h, w)
            return

        imageBack = gdrawable.get_image(x, y, w, h)
        imageVert = gdrawable.get_image(x, y, h, w)
        imageFlip = gtk.gdk.Image(type=gdk.IMAGE_FASTEST,
                                  visual=gdrawable.get_visual(),
                                  width=w, height=h)
        if imageFlip == None or imageBack == None or imageVert == None:
            warnings.warn("Could not renderer vertical text")
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

        Note that pango assumes a logical DPI of 96
        Ref: pango/fonts.c/pango_font_description_set_size() manual page
        """
        # problem? - cache gets bigger and bigger, is never cleared out
        # two (not one) layouts are created for every text item s (then they
        # are cached) - why?

        key = self.dpi.get(), s, hash(prop)
        value = self.layoutd.get(key)
        if value != None:
            return value

        size = prop.get_size_in_points() * self.dpi.get() / 96.0
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
        return points/72.0 * self.dpi.get()


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


    #def set_antialiased(self, b):
        # anti-aliasing is not supported by GDK

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
        if w == 0:
            self.gdkGC.line_width = 0
        else:
            pixels = self.renderer.points_to_pixels(w)
            self.gdkGC.line_width = max(1, int(round(pixels)))


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas  = FigureCanvasGDK(thisFig)
    manager = FigureManagerBase(canvas, num)
    # equals:
    #manager = FigureManagerBase (FigureCanvasGDK (Figure(*args, **kwargs),
    #                             num)
    return manager


class FigureCanvasGDK (FigureCanvasBase):
    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)

        self._renderer_init()

    def _renderer_init(self):
        self._renderer = RendererGDK (gtk.DrawingArea(), self.figure.dpi)

    def _render_figure(self, pixmap, width, height):
        self._renderer.set_pixmap (pixmap)
        self._renderer.set_width_height (width, height)
        self.figure.draw (self._renderer)

    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait', **kwargs):
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
            width, height = self.get_width_height()
            pixmap = gtk.gdk.Pixmap (None, width, height, depth=24)
            self._render_figure(pixmap, width, height)

            # jpg colors don't match the display very well, png colors match
            # better
            pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, 0, 8,
                                    width, height)
            pixbuf.get_from_drawable(pixmap, pixmap.get_colormap(),
                                     0, 0, 0, 0, width, height)

            # pixbuf.save() recognises 'jpeg' not 'jpg'
            if ext == 'jpg': ext = 'jpeg'

            pixbuf.save(filename, ext)

        elif ext in ('eps', 'ps', 'svg',):
            if ext == 'svg':
                from backend_svg import FigureCanvasSVG as FigureCanvas
            else:
                from backend_ps  import FigureCanvasPS  as FigureCanvas


            fc = self.switch_backends(FigureCanvas)
            fc.print_figure(filename, dpi, facecolor, edgecolor, orientation,
                            **kwargs)
        elif ext in ('bmp', 'raw', 'rgb',):

            from backend_agg import FigureCanvasAgg  as FigureCanvas
            fc = self.switch_backends(FigureCanvas)
            fc.print_figure(filename, dpi, facecolor, edgecolor, orientation,
                            **kwargs)

        else:
            raise ValueError('Format "%s" is not supported.\nSupported formats are %s.' %
                             (ext, ', '.join(IMAGE_FORMAT)))

        self.figure.set_canvas(self)
