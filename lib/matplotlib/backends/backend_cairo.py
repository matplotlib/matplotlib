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


class ArrayWrapper:
    """Thin wrapper around numpy ndarray to expose the interface
       expected by cairocffi. Basically replicates the
       array.array interface.
    """
    def __init__(self, myarray):
        self.__array = myarray
        self.__data = myarray.ctypes.data
        self.__size = len(myarray.flatten())
        self.itemsize = myarray.itemsize

    def buffer_info(self):
        return (self.__data, self.__size)


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
    def convert_path(ctx, path, transform, clip=None):
        for points, code in path.iter_segments(transform, clip=clip):
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
        ctx = gc.ctx

        # We'll clip the path to the actual rendering extents
        # if the path isn't filled.
        if rgbFace is None and gc.get_hatch() is None:
            clip = ctx.clip_extents()
        else:
            clip = None

        transform = transform + \
            Affine2D().scale(1.0, -1.0).translate(0, self.height)

        ctx.new_path()
        self.convert_path(ctx, path, transform, clip)

        self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_markers(self, gc, marker_path, marker_trans, path, transform, rgbFace=None):
        ctx = gc.ctx

        ctx.new_path()
        # Create the path for the marker; it needs to be flipped here already!
        self.convert_path(ctx, marker_path, marker_trans + Affine2D().scale(1.0, -1.0))
        marker_path = ctx.copy_path_flat()

        # Figure out whether the path has a fill
        x1, y1, x2, y2 = ctx.fill_extents()
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            filled = False
            # No fill, just unset this (so we don't try to fill it later on)
            rgbFace = None
        else:
            filled = True

        transform = transform + \
            Affine2D().scale(1.0, -1.0).translate(0, self.height)

        ctx.new_path()
        for i, (vertices, codes) in enumerate(path.iter_segments(transform, simplify=False)):
            if len(vertices):
                x, y = vertices[-2:]
                ctx.save()

                # Translate and apply path
                ctx.translate(x, y)
                ctx.append_path(marker_path)

                ctx.restore()

                # Slower code path if there is a fill; we need to draw
                # the fill and stroke for each marker at the same time.
                # Also flush out the drawing every once in a while to
                # prevent the paths from getting way too long.
                if filled or i % 1000 == 0:
                    self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

        # Fast path, if there is no fill, draw everything in one step
        if not filled:
            self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_image(self, gc, x, y, im):
        # bbox - not currently used
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))

        if sys.byteorder == 'little':
            im = im[:, :, (2, 1, 0, 3)]
        else:
            im = im[:, :, (3, 0, 1, 2)]
        if HAS_CAIRO_CFFI:
            # cairocffi tries to use the buffer_info from array.array
            # that we replicate in ArrayWrapper and alternatively falls back
            # on ctypes to get a pointer to the numpy array. This works
            # correctly on a numpy array in python3 but not 2.7. We replicate
            # the array.array functionality here to get cross version support.
            imbuffer = ArrayWrapper(im.flatten())
        else:
            # py2cairo uses PyObject_AsWriteBuffer
            # to get a pointer to the numpy array this works correctly
            # on a regular numpy array but not on a memory view.
            # At the time of writing the latest release version of
            # py3cairo still does not support create_for_data
            imbuffer = im.flatten()
        surface = cairo.ImageSurface.create_for_data(imbuffer,
                                                     cairo.FORMAT_ARGB32,
                                                     im.shape[1],
                                                     im.shape[0],
                                                     im.shape[1]*4)
        ctx = gc.ctx
        y = self.height - y - im.shape[0]

        ctx.save()
        ctx.set_source_surface(surface, float(x), float(y))
        if gc.get_alpha() != 1.0:
            ctx.paint_with_alpha(gc.get_alpha())
        else:
            ctx.paint()
        ctx.restore()

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
                ctx.rotate(np.deg2rad(-angle))
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
            ctx.rotate(np.deg2rad(-angle))

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
            if not six.PY3 and isinstance(s, six.text_type):
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
        x,y,w,h = np.round(x), np.round(y), np.round(w), np.round(h)
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

    def get_rgb(self):
        return self.ctx.get_source().get_rgba()[:3]

    def set_joinstyle(self, js):
        if js in ('miter', 'round', 'bevel'):
            self._joinstyle = js
            self.ctx.set_line_join(self._joind[js])
        else:
            raise ValueError('Unrecognized join style.  Found %s' % js)


    def set_linewidth(self, w):
        self._linewidth = float(w)
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
                if is_string_like(fo):
                    fo = gzip.GzipFile(fo, 'wb')
                else:
                    fo = gzip.GzipFile(None, 'wb', fileobj=fo)
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
        if format == 'svgz':
            fo.close()


FigureCanvas = FigureCanvasCairo
