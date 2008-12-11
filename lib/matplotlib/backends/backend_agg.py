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

  * integrate screen dpi w/ ppi and text
"""
from __future__ import division

import numpy as npy

from matplotlib import verbose, rcParams
from matplotlib.backend_bases import RendererBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import is_string_like, maxdict
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
from matplotlib.ft2font import FT2Font, LOAD_FORCE_AUTOHINT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox

from _backend_agg import RendererAgg as _RendererAgg
from matplotlib import _png

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
        if __debug__: verbose.report('RendererAgg.__init__ width=%s, height=%s'%(width, height), 'debug-annoying')
        self._renderer = _RendererAgg(int(width), int(height), dpi, debug=False)
        if __debug__: verbose.report('RendererAgg.__init__ _RendererAgg done',
                                     'debug-annoying')
        #self.draw_path = self._renderer.draw_path  # see below
        self.draw_markers = self._renderer.draw_markers
        self.draw_path_collection = self._renderer.draw_path_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.draw_image = self._renderer.draw_image
        self.copy_from_bbox = self._renderer.copy_from_bbox
        self.restore_region = self._renderer.restore_region
        self.tostring_rgba_minimized = self._renderer.tostring_rgba_minimized
        self.mathtext_parser = MathTextParser('Agg')

        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)
        if __debug__: verbose.report('RendererAgg.__init__ done',
                                     'debug-annoying')

    def draw_path(self, gc, path, transform, rgbFace=None):
        nmax = rcParams['agg.path.chunksize'] # here at least for testing
        npts = path.vertices.shape[0]
        if nmax > 100 and npts > nmax and path.should_simplify and rgbFace is None:
            nch = npy.ceil(npts/float(nmax))
            chsize = int(npy.ceil(npts/nch))
            i0 = npy.arange(0, npts, chsize)
            i1 = npy.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for ii0, ii1 in zip(i0, i1):
                v = path.vertices[ii0:ii1,:]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO # move to end of last chunk
                p = Path(v, c)
                self._renderer.draw_path(gc, p, transform, rgbFace)
        else:
            self._renderer.draw_path(gc, path, transform, rgbFace)


    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """
        Draw the math text using matplotlib.mathtext
        """
        if __debug__: verbose.report('RendererAgg.draw_mathtext',
                                     'debug-annoying')
        ox, oy, width, height, descent, font_image, used_characters = \
            self.mathtext_parser.parse(s, self.dpi, prop)

        x = int(x) + ox
        y = int(y) - oy
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

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
            # We pass '0' for angle here, since it will be rotated (in raster
            # space) in the following call to draw_text_image).
            font.set_text(s, 0, flags=LOAD_FORCE_AUTOHINT)
        font.draw_glyphs_to_bitmap()

        #print x, y, int(x), int(y)

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
            Z = texmanager.get_grey(s, size, self.dpi)
            m,n = Z.shape
            # TODO: descent of TeX text (I am imitating backend_ps here -JKS)
            return n, m, 0

        if ismath:
            ox, oy, width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi, prop)
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

        texmanager = self.get_texmanager()
        key = s, size, self.dpi, angle, texmanager.get_font_config()
        im = self.texd.get(key)
        if im is None:
            Z = texmanager.get_grey(s, size, self.dpi)
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
        font.set_size(size, self.dpi)

        return font

    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        if __debug__: verbose.report('RendererAgg.points_to_pixels',
                                     'debug-annoying')
        return points*self.dpi/72.0

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

    def option_image_nocomposite(self):
        # It is generally faster to composite each image directly to
        # the Figure, and there's no file size benefit to compositing
        # with the Agg backend
        return True


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
        l, b, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi
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

    def print_raw(self, filename_or_obj, *args, **kwargs):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = self.figure.dpi
        if is_string_like(filename_or_obj):
            filename_or_obj = file(filename_or_obj, 'wb')
        renderer._renderer.write_rgba(filename_or_obj)
        renderer.dpi = original_dpi
    print_rgba = print_raw

    def print_png(self, filename_or_obj, *args, **kwargs):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = self.figure.dpi
        if is_string_like(filename_or_obj):
            filename_or_obj = file(filename_or_obj, 'wb')
        _png.write_png(renderer._renderer.buffer_rgba(0, 0),
                       renderer.width, renderer.height,
                       filename_or_obj, self.figure.dpi)
        renderer.dpi = original_dpi
