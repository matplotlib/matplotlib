"""
An agg http://antigrain.com/ backend

Features that are implemented

 * capstyles and join styles
 * dashes
 * linewidth
 * lines, rectangles, ellipses
 * clipping to a rectangle
 * output to RGBA and PNG
 * alpha blending
 * DPI scaling properly - everything scales properly (dashes, linewidths, etc)
 * draw polygon
 * freetype2 w/ ft2font

TODO:

  * allow save to file handle

  * allow load from png

  * integrate screen dpi w/ ppi and text

INSTALLING

  REQUIREMENTs

    python2.3+
    numpy 1.0 +

    agg2 (see below)
    freetype 2
    libpng
    libz

  Install AGG2 (cut and paste below into xterm should work)

    wget http://www.antigrain.com/agg2.tar.gz
    tar xvfz agg2.tar.gz
    cd agg2
    make

    (Optional) if you want to make the examples:
    cd examples/X11
    make

  Installing backend_agg


   Edit setup.py: change aggsrc to point to the agg2 src tree and
   replace if 0: with if 1: in the backend_agg section

   Then just do the usual thing: python setup.py build

   Please let me know if you encounter build problems, and tell me
   platform, gcc version, etc...  Currently the paths in setupext.py
   assume as linux like filesystem (eg X11 include dir, location of
   libttf, etcc) so you may need to tweak these

  Using agg backend

    python somefile.py -dAgg

  or

    import matplotlib
    matplotlib.use('Agg')


"""
from __future__ import division
import os, sys

import numpy as npy

import matplotlib
from matplotlib import verbose, rcParams
from matplotlib._image import fromarray
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase,\
     GraphicsContextBase, FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import enumerate, is_string_like, exception_to_str, \
    maxdict
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
from matplotlib.ft2font import FT2Font, LOAD_FORCE_AUTOHINT
from matplotlib.mathtext import MathTextParser
from matplotlib.transforms import lbwh_to_bbox

from _backend_agg import RendererAgg as _RendererAgg

backend_version = 'v2.2'

class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    debug=1
    texd = maxdict(50)  # a cache of tex image rasters
    _fontd = maxdict(50)
    def __init__(self, width, height, dpi):
        if __debug__: verbose.report('RendererAgg.__init__', 'debug-annoying')
        RendererBase.__init__(self)
        self.dpi = dpi
        self.width = width
        self.height = height
        if __debug__: verbose.report('RendererAgg.__init__ width=%s, \
                        height=%s'%(width, height), 'debug-annoying')
        self._renderer = _RendererAgg(int(width), int(height), dpi.get(),
                                    debug=False)
        if __debug__: verbose.report('RendererAgg.__init__ _RendererAgg done',
                                     'debug-annoying')
        self.draw_polygon = self._renderer.draw_polygon
        self.draw_rectangle = self._renderer.draw_rectangle
        self.draw_path = self._renderer.draw_path
        self.draw_lines = self._renderer.draw_lines
        self.draw_markers = self._renderer.draw_markers
        self.draw_image = self._renderer.draw_image
        self.draw_line_collection = self._renderer.draw_line_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.draw_poly_collection = self._renderer.draw_poly_collection
        self.draw_regpoly_collection = self._renderer.draw_regpoly_collection

        self.copy_from_bbox = self._renderer.copy_from_bbox
        self.restore_region = self._renderer.restore_region
        self.mathtext_parser = MathTextParser('Agg')

        self.bbox = lbwh_to_bbox(0,0, self.width, self.height)
        if __debug__: verbose.report('RendererAgg.__init__ done',
                                     'debug-annoying')

    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2, rotation):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0

        If rgbFace is not None, fill the rectangle with that color.  gcEdge
        is a GraphicsContext instance

        Currently, I'm only supporting ellipses, ie angle args are
        ignored
        """
        if __debug__: verbose.report('RendererAgg.draw_arc', 'debug-annoying')
        self._renderer.draw_ellipse(
            gcEdge, rgbFace, x, y, width/2., height/2., rotation)  # ellipse takes radius


    def draw_line(self, gc, x1, y1, x2, y2):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if __debug__: verbose.report('RendererAgg.draw_line', 'debug-annoying')
        x = npy.array([x1,x2], float)
        y = npy.array([y1,y2], float)
        self._renderer.draw_lines(gc, x, y)


    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        if __debug__: verbose.report('RendererAgg.draw_point', 'debug-annoying')
        rgbFace = gc.get_rgb()
        self._renderer.draw_ellipse(
            gc, rgbFace, x, y, 0.5, 0.5, 0.0)

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """
        Draw the math text using matplotlib.mathtext
        """
        if __debug__: verbose.report('RendererAgg.draw_mathtext',
                                     'debug-annoying')
        ox, oy, width, height, descent, font_image, used_characters = \
            self.mathtext_parser.parse(s, self.dpi.get(), prop)

        x = int(x) + ox
        y = int(y) - oy
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)
        if 0:
            self._renderer.draw_rectangle(gc, None,
                                          int(x),
                                          self.height-int(y),
                                          width, height)

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        """
        Render the text
        """
        if __debug__: verbose.report('RendererAgg.draw_text', 'debug-annoying')

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        font = self._get_agg_font(prop)
        if font is None: return None
        if len(s) == 1 and ord(s) > 127:
            font.load_char(ord(s), flags=LOAD_FORCE_AUTOHINT)
        else:
            font.set_text(s, 0, flags=LOAD_FORCE_AUTOHINT)
        font.draw_glyphs_to_bitmap()

        #print x, y, int(x), int(y)

        # We pass '0' for angle here, since is has already been rotated
        # (in vector space) in the above call to font.set_text.
        self._renderer.draw_text_image(font.get_image(), int(x), int(y) + 1, angle, gc)


    def get_text_width_height_descent(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop

        # passing rgb is a little hack to make cacheing in the
        # texmanager more efficient.  It is not meant to be used
        # outside the backend
        """
        if ismath=='TeX':
            # todo: handle props
            size = prop.get_size_in_points()
            texmanager = self.get_texmanager()
            Z = texmanager.get_grey(s, size, self.dpi.get())
            m,n = Z.shape
            # TODO: descent of TeX text (I am imitating backend_ps here -JKS)
            return n, m, 0

        if ismath:
            ox, oy, width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi.get(), prop)
            return width, height, descent
        font = self._get_agg_font(prop)
        font.set_text(s, 0.0, flags=LOAD_FORCE_AUTOHINT)  # the width and height of unrotated string
        w, h = font.get_width_height()
        d = font.get_descent()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d /= 64.0
        return w, h, d

    def draw_tex(self, gc, x, y, s, prop, angle):
        # todo, handle props, angle, origins
        size = prop.get_size_in_points()
        dpi = self.dpi.get()

        texmanager = self.get_texmanager()
        key = s, size, dpi, angle, texmanager.get_font_config()
        im = self.texd.get(key)
        if im is None:
            Z = texmanager.get_grey(s, size, dpi)
            Z = npy.array(Z * 255.0, npy.uint8)

        self._renderer.draw_text_image(Z, x, y, angle, gc)


    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height


    def _get_agg_font(self, prop):
        """
        Get the font for text instance t, cacheing for efficiency
        """
        if __debug__: verbose.report('RendererAgg._get_agg_font',
                                     'debug-annoying')

        key = hash(prop)
        font = self._fontd.get(key)

        if font is None:
            fname = findfont(prop)
            font = self._fontd.get(fname)
            if font is None:
                font = FT2Font(str(fname))
                self._fontd[fname] = font
            self._fontd[key] = font

        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, self.dpi.get())

        return font


    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        if __debug__: verbose.report('RendererAgg.points_to_pixels',
                                     'debug-annoying')
        return points*self.dpi.get()/72.0

    def tostring_rgb(self):
        if __debug__: verbose.report('RendererAgg.tostring_rgb',
                                     'debug-annoying')
        return self._renderer.tostring_rgb()

    def tostring_argb(self):
        if __debug__: verbose.report('RendererAgg.tostring_argb',
                                     'debug-annoying')
        return self._renderer.tostring_argb()

    def buffer_rgba(self,x,y):
        if __debug__: verbose.report('RendererAgg.buffer_rgba',
                                     'debug-annoying')
        return self._renderer.buffer_rgba(x,y)

    def clear(self):
        self._renderer.clear()



def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if __debug__: verbose.report('backend_agg.new_figure_manager',
                                 'debug-annoying')


    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasAgg(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


class FigureCanvasAgg(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
    """

    def copy_from_bbox(self, bbox):
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region):
        renderer = self.get_renderer()
        return renderer.restore_region(region)

    def draw(self):
        """
        Draw the figure using the renderer
        """
        if __debug__: verbose.report('FigureCanvasAgg.draw', 'debug-annoying')

        self.renderer = self.get_renderer()
        self.figure.draw(self.renderer)

    def get_renderer(self):
        l,b,w,h = self.figure.bbox.get_bounds()
        key = w, h, self.figure.dpi.get()
        try: self._lastKey, self.renderer
        except AttributeError: need_new_renderer = True
        else:  need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        return self.renderer

    def tostring_rgb(self):
        if __debug__: verbose.report('FigureCanvasAgg.tostring_rgb',
                                     'debug-annoying')
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        if __debug__: verbose.report('FigureCanvasAgg.tostring_argb',
                                     'debug-annoying')
        return self.renderer.tostring_argb()

    def buffer_rgba(self,x,y):
        if __debug__: verbose.report('FigureCanvasAgg.buffer_rgba',
                                     'debug-annoying')
        return self.renderer.buffer_rgba(x,y)

    def get_default_filetype(self):
        return 'png'

    def print_raw(self, filename, *args, **kwargs):
        self.draw()
        self.get_renderer()._renderer.write_rgba(str(filename))
    print_rgba = print_raw

    def print_png(self, filename, *args, **kwargs):
        self.draw()
        self.get_renderer()._renderer.write_png(filename, self.figure.dpi.get())

