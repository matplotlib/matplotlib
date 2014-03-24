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
    cairo, pycairo

Naming Conventions
  * classes MixedUpperCase
  * varables lowerUpper
  * functions underscore_separated
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os, sys, warnings, gzip

import numpy as np

def _fn_name(): return sys._getframe(1).f_code.co_name

try:
    import cairocffi as cairo
except ImportError:
    try:
        import cairo
    except ImportError:
        raise ImportError("Cairo backend requires that cairocffi or pycairo is installed.")
    else:
        HAS_CAIRO_CFFI = False
else:
    HAS_CAIRO_CFFI = True

_version_required = (1,2,0)
if cairo.version_info < _version_required:
    raise ImportError ("Pycairo %d.%d.%d is installed\n"
                       "Pycairo %d.%d.%d or later is required"
                        % (cairo.version_info + _version_required))
backend_version = cairo.version
del _version_required

from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook        import is_string_like
from matplotlib.figure       import Figure
from matplotlib.mathtext     import MathTextParser
from matplotlib.path         import Path
from matplotlib.transforms   import Bbox, Affine2D
from matplotlib.font_manager import ttfFontProperty

_debug = False
#_debug = True

# Image::color_conv(format) for draw_image()
if sys.byteorder == 'little':
    BYTE_FORMAT = 0 # BGRA
else:
    BYTE_FORMAT = 1 # ARGB


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
        """
        """
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        self.dpi = dpi
        self.gc = GraphicsContextCairo (renderer=self)
        self.text_ctx = cairo.Context (
           cairo.ImageSurface (cairo.FORMAT_ARGB32,1,1))
        self.mathtext_parser = MathTextParser('Cairo')

        RendererBase.__init__(self)

    def set_ctx_from_surface (self, surface):
        self.gc.ctx = cairo.Context (surface)


    def set_width_height(self, width, height):
        self.width  = width
        self.height = height
        self.matrix_flipy = cairo.Matrix (yy=-1, y0=self.height)
        # use matrix_flipy for ALL rendering?
        # - problem with text? - will need to switch matrix_flipy off, or do a
        # font transform?


    def _fill_and_stroke (self, ctx, fill_c, alpha, alpha_overrides):
        if fill_c is not None:
            ctx.save()
            if len(fill_c) == 3 or alpha_overrides:
                ctx.set_source_rgba (fill_c[0], fill_c[1], fill_c[2], alpha)
            else:
                ctx.set_source_rgba (fill_c[0], fill_c[1], fill_c[2], fill_c[3])
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()

    @staticmethod
    def convert_path(ctx, path, transform):
        for points, code in path.iter_segments(transform):
            if code == Path.MOVETO:
                ctx.move_to(*points)
            elif code == Path.CLOSEPOLY:
                ctx.close_path()
            elif code == Path.LINETO:
                ctx.line_to(*points)
            elif code == Path.CURVE3:
                ctx.curve_to(points[0], points[1],
                             points[0], points[1],
                             points[2], points[3])
            elif code == Path.CURVE4:
                ctx.curve_to(*points)


    def draw_path(self, gc, path, transform, rgbFace=None):
        if len(path.vertices) > 18980:
            raise ValueError("The Cairo backend can not draw paths longer than 18980 points.")

        ctx = gc.ctx

        transform = transform + \
            Affine2D().scale(1.0, -1.0).translate(0, self.height)

        ctx.new_path()
        self.convert_path(ctx, path, transform)

        self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_image(self, gc, x, y, im):
        # bbox - not currently used
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))

        im.flipud_out()

        rows, cols, buf = im.color_conv (BYTE_FORMAT)
        surface = cairo.ImageSurface.create_for_data (
                      buf, cairo.FORMAT_ARGB32, cols, rows, cols*4)
        ctx = gc.ctx
        y = self.height - y - rows

        ctx.save()
        ctx.set_source_surface (surface, x, y)
        if gc.get_alpha() != 1.0:
            ctx.paint_with_alpha(gc.get_alpha())
        else:
            ctx.paint()
        ctx.restore()

        im.flipud_out()

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # Note: x,y are device/display coords, not user-coords, unlike other
        # draw_* methods
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))

        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)

        else:
            ctx = gc.ctx
            ctx.new_path()
            ctx.move_to (x, y)
            ctx.select_font_face (prop.get_name(),
                                  self.fontangles [prop.get_style()],
                                  self.fontweights[prop.get_weight()])

            size = prop.get_size_in_points() * self.dpi / 72.0

            ctx.save()
            if angle:
                ctx.rotate (-angle * np.pi / 180)
            ctx.set_font_size (size)

            if HAS_CAIRO_CFFI:
                if not isinstance(s, six.text_type):
                    s = six.text_type(s)
            else:
                if not six.PY3 and isinstance(s, six.text_type):
                    s = s.encode("utf-8")

            ctx.show_text(s)
            ctx.restore()

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))

        ctx = gc.ctx
        width, height, descent, glyphs, rects = self.mathtext_parser.parse(
            s, self.dpi, prop)

        ctx.save()
        ctx.translate(x, y)
        if angle:
            ctx.rotate (-angle * np.pi / 180)

        for font, fontsize, s, ox, oy in glyphs:
            ctx.new_path()
            ctx.move_to(ox, oy)

            fontProp = ttfFontProperty(font)
            ctx.save()
            ctx.select_font_face (fontProp.name,
                                  self.fontangles [fontProp.style],
                                  self.fontweights[fontProp.weight])

            size = fontsize * self.dpi / 72.0
            ctx.set_font_size(size)
            if isinstance(s, six.text_type):
                s = s.encode("utf-8")
            ctx.show_text(s)
            ctx.restore()

        for ox, oy, w, h in rects:
            ctx.new_path()
            ctx.rectangle (ox, oy, w, h)
            ctx.set_source_rgb (0, 0, 0)
            ctx.fill_preserve()

        ctx.restore()


    def flipy(self):
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        return True
        #return False # tried - all draw objects ok except text (and images?)
        # which comes out mirrored!


    def get_canvas_width_height(self):
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        return self.width, self.height


    def get_text_width_height_descent(self, s, prop, ismath):
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        if ismath:
            width, height, descent, fonts, used_characters = self.mathtext_parser.parse(
               s, self.dpi, prop)
            return width, height, descent

        ctx = self.text_ctx
        ctx.save()
        ctx.select_font_face (prop.get_name(),
                              self.fontangles [prop.get_style()],
                              self.fontweights[prop.get_weight()])

        # Cairo (says it) uses 1/96 inch user space units, ref: cairo_gstate.c
        # but if /96.0 is used the font is too small

        size = prop.get_size_in_points() * self.dpi / 72.0

        # problem - scale remembers last setting and font can become
        # enormous causing program to crash
        # save/restore prevents the problem
        ctx.set_font_size (size)

        y_bearing, w, h = ctx.text_extents (s)[1:4]
        ctx.restore()

        return w, h, h + y_bearing


    def new_gc(self):
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        self.gc.ctx.save()
        self.gc._alpha = 1.0
        self.gc._forced_alpha = False # if True, _alpha overrides A from RGBA
        return self.gc


    def points_to_pixels(self, points):
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        return points/72.0 * self.dpi


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


    def restore(self):
        self.ctx.restore()


    def set_alpha(self, alpha):
        GraphicsContextBase.set_alpha(self, alpha)
        _alpha = self.get_alpha()
        rgb = self._rgb
        if self.get_forced_alpha():
            self.ctx.set_source_rgba (rgb[0], rgb[1], rgb[2], _alpha)
        else:
            self.ctx.set_source_rgba (rgb[0], rgb[1], rgb[2], rgb[3])


    #def set_antialiased(self, b):
        # enable/disable anti-aliasing is not (yet) supported by Cairo


    def set_capstyle(self, cs):
        if cs in ('butt', 'round', 'projecting'):
            self._capstyle = cs
            self.ctx.set_line_cap (self._capd[cs])
        else:
            raise ValueError('Unrecognized cap style.  Found %s' % cs)


    def set_clip_rectangle(self, rectangle):
        if not rectangle: return
        x,y,w,h = rectangle.bounds
        # pixel-aligned clip-regions are faster
        x,y,w,h = round(x), round(y), round(w), round(h)
        ctx = self.ctx
        ctx.new_path()
        ctx.rectangle (x, self.renderer.height - h - y, w, h)
        ctx.clip ()

    def set_clip_path(self, path):
        if not path: return
        tpath, affine = path.get_transformed_path_and_affine()
        ctx = self.ctx
        ctx.new_path()
        affine = affine + Affine2D().scale(1.0, -1.0).translate(0.0, self.renderer.height)
        RendererCairo.convert_path(ctx, tpath, affine)
        ctx.clip()

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes == None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            self.ctx.set_dash(
                list(self.renderer.points_to_pixels(np.asarray(dashes))), offset)


    def set_foreground(self, fg, isRGBA=None):
        GraphicsContextBase.set_foreground(self, fg, isRGBA)
        if len(self._rgb) == 3:
            self.ctx.set_source_rgb(*self._rgb)
        else:
            self.ctx.set_source_rgba(*self._rgb)

    def set_graylevel(self, frac):
        GraphicsContextBase.set_graylevel(self, frac)
        if len(self._rgb) == 3:
            self.ctx.set_source_rgb(*self._rgb)
        else:
            self.ctx.set_source_rgba(*self._rgb)


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
    if _debug: print('%s()' % (_fn_name()))
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas  = FigureCanvasCairo(figure)
    manager = FigureManagerBase(canvas, num)
    return manager


class FigureCanvasCairo (FigureCanvasBase):
    def print_png(self, fobj, *args, **kwargs):
        width, height = self.get_width_height()

        renderer = RendererCairo (self.figure.dpi)
        renderer.set_width_height (width, height)
        surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, width, height)
        renderer.set_ctx_from_surface (surface)

        self.figure.draw (renderer)
        surface.write_to_png (fobj)

    def print_pdf(self, fobj, *args, **kwargs):
        return self._save(fobj, 'pdf', *args, **kwargs)

    def print_ps(self, fobj, *args, **kwargs):
        return self._save(fobj, 'ps', *args, **kwargs)

    def print_svg(self, fobj, *args, **kwargs):
        return self._save(fobj, 'svg', *args, **kwargs)

    def print_svgz(self, fobj, *args, **kwargs):
        return self._save(fobj, 'svgz', *args, **kwargs)

    def _save (self, fo, format, **kwargs):
        # save PDF/PS/SVG
        orientation = kwargs.get('orientation', 'portrait')

        dpi = 72
        self.figure.dpi = dpi
        w_in, h_in = self.figure.get_size_inches()
        width_in_points, height_in_points = w_in * dpi, h_in * dpi

        if orientation == 'landscape':
            width_in_points, height_in_points = (height_in_points,
                                                 width_in_points)

        if format == 'ps':
            if not hasattr(cairo, 'PSSurface'):
                raise RuntimeError ('cairo has not been compiled with PS '
                                    'support enabled')
            surface = cairo.PSSurface (fo, width_in_points, height_in_points)
        elif format == 'pdf':
            if not hasattr(cairo, 'PDFSurface'):
                raise RuntimeError ('cairo has not been compiled with PDF '
                                    'support enabled')
            surface = cairo.PDFSurface (fo, width_in_points, height_in_points)
        elif format in ('svg', 'svgz'):
            if not hasattr(cairo, 'SVGSurface'):
                raise RuntimeError ('cairo has not been compiled with SVG '
                                    'support enabled')
            if format == 'svgz':
                filename = fo
                if is_string_like(fo):
                    fo = open(fo, 'wb')
                    close = True
                else:
                    close = False
                try:
                    fo = gzip.GzipFile(None, 'wb', fileobj=fo)
                finally:
                    if close:
                        fo.close()
            surface = cairo.SVGSurface (fo, width_in_points, height_in_points)
        else:
           warnings.warn ("unknown format: %s" % format)
           return

        # surface.set_dpi() can be used
        renderer = RendererCairo (self.figure.dpi)
        renderer.set_width_height (width_in_points, height_in_points)
        renderer.set_ctx_from_surface (surface)
        ctx = renderer.gc.ctx

        if orientation == 'landscape':
            ctx.rotate (np.pi/2)
            ctx.translate (0, -height_in_points)
            # cairo/src/cairo_ps_surface.c
            # '%%Orientation: Portrait' is always written to the file header
            # '%%Orientation: Landscape' would possibly cause problems
            # since some printers would rotate again ?
            # TODO:
            # add portrait/landscape checkbox to FileChooser

        self.figure.draw (renderer)

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
        surface.finish()


FigureCanvas = FigureCanvasCairo
