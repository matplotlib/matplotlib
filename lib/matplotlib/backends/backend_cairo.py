"""
A Cairo backend for matplotlib
==============================
:Author: Steve Chaplin and others

This backend depends on cairocffi or pycairo.
"""

import copy
import gzip
import sys
import warnings

import numpy as np

# cairocffi is more widely compatible than pycairo (in particular pgi only
# works with cairocffi) so try it first.
try:
    import cairocffi as cairo
except ImportError:
    try:
        import cairo
    except ImportError:
        raise ImportError("cairo backend requires that cairocffi or pycairo "
                          "is installed")
    else:
        if cairo.version_info < (1, 11, 0):
            # Introduced create_for_data for Py3.
            raise ImportError(
                "cairo {} is installed; cairo>=1.11.0 is required"
                .format(cairo.version))

backend_version = cairo.version

from .. import cbook
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
    RendererBase)
from matplotlib.font_manager import ttfFontProperty
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


if cairo.__name__ == "cairocffi":
    # Convert a pycairo context to a cairocffi one.
    def _to_context(ctx):
        if not isinstance(ctx, cairo.Context):
            ctx = cairo.Context._from_pointer(
                cairo.ffi.cast(
                    'cairo_t **',
                    id(ctx) + object.__basicsize__)[0],
                incref=True)
        return ctx
else:
    # Pass-through a pycairo context.
    def _to_context(ctx):
        return ctx


@cbook.deprecated("3.0")
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


# Mapping from Matplotlib Path codes to cairo path codes.
_MPL_TO_CAIRO_PATH_TYPE = np.zeros(80, dtype=int)  # CLOSEPOLY = 79.
_MPL_TO_CAIRO_PATH_TYPE[Path.MOVETO] = cairo.PATH_MOVE_TO
_MPL_TO_CAIRO_PATH_TYPE[Path.LINETO] = cairo.PATH_LINE_TO
_MPL_TO_CAIRO_PATH_TYPE[Path.CURVE4] = cairo.PATH_CURVE_TO
_MPL_TO_CAIRO_PATH_TYPE[Path.CLOSEPOLY] = cairo.PATH_CLOSE_PATH
# Sizes in cairo_path_data_t of each cairo path element.
_CAIRO_PATH_TYPE_SIZES = np.zeros(4, dtype=int)
_CAIRO_PATH_TYPE_SIZES[cairo.PATH_MOVE_TO] = 2
_CAIRO_PATH_TYPE_SIZES[cairo.PATH_LINE_TO] = 2
_CAIRO_PATH_TYPE_SIZES[cairo.PATH_CURVE_TO] = 4
_CAIRO_PATH_TYPE_SIZES[cairo.PATH_CLOSE_PATH] = 1


def _append_paths_slow(ctx, paths, transforms, clip=None):
    for path, transform in zip(paths, transforms):
        for points, code in path.iter_segments(transform, clip=clip):
            if code == Path.MOVETO:
                ctx.move_to(*points)
            elif code == Path.CLOSEPOLY:
                ctx.close_path()
            elif code == Path.LINETO:
                ctx.line_to(*points)
            elif code == Path.CURVE3:
                cur = ctx.get_current_point()
                ctx.curve_to(
                    *np.concatenate([cur / 3 + points[:2] * 2 / 3,
                                     points[:2] * 2 / 3 + points[-2:] / 3]))
            elif code == Path.CURVE4:
                ctx.curve_to(*points)


def _append_paths_fast(ctx, paths, transforms, clip=None):
    # We directly convert to the internal representation used by cairo, for
    # which ABI compatibility is guaranteed.  The layout for each item is
    # --CODE(4)--  -LENGTH(4)-  ---------PAD(8)---------
    # ----------X(8)----------  ----------Y(8)----------
    # with the size in bytes in parentheses, and (X, Y) repeated as many times
    # as there are points for the current code.
    ffi = cairo.ffi

    # Convert curves to segment, so that 1. we don't have to handle
    # variable-sized CURVE-n codes, and 2. we don't have to implement degree
    # elevation for quadratic Beziers.
    cleaneds = [path.cleaned(transform=transform, clip=clip, curves=False)
                for path, transform in zip(paths, transforms)]
    vertices = np.concatenate([cleaned.vertices for cleaned in cleaneds])
    codes = np.concatenate([cleaned.codes for cleaned in cleaneds])

    # Remove unused vertices and convert to cairo codes.  Note that unlike
    # cairo_close_path, we do not explicitly insert an extraneous MOVE_TO after
    # CLOSE_PATH, so our resulting buffer may be smaller.
    vertices = vertices[(codes != Path.STOP) & (codes != Path.CLOSEPOLY)]
    codes = codes[codes != Path.STOP]
    codes = _MPL_TO_CAIRO_PATH_TYPE[codes]

    # Where are the headers of each cairo portions?
    cairo_type_sizes = _CAIRO_PATH_TYPE_SIZES[codes]
    cairo_type_positions = np.insert(np.cumsum(cairo_type_sizes), 0, 0)
    cairo_num_data = cairo_type_positions[-1]
    cairo_type_positions = cairo_type_positions[:-1]

    # Fill the buffer.
    buf = np.empty(cairo_num_data * 16, np.uint8)
    as_int = np.frombuffer(buf.data, np.int32)
    as_int[::4][cairo_type_positions] = codes
    as_int[1::4][cairo_type_positions] = cairo_type_sizes
    as_float = np.frombuffer(buf.data, np.float64)
    mask = np.ones_like(as_float, bool)
    mask[::2][cairo_type_positions] = mask[1::2][cairo_type_positions] = False
    as_float[mask] = vertices.ravel()

    # Construct the cairo_path_t, and pass it to the context.
    ptr = ffi.new("cairo_path_t *")
    ptr.status = cairo.STATUS_SUCCESS
    ptr.data = ffi.cast("cairo_path_data_t *", ffi.from_buffer(buf))
    ptr.num_data = cairo_num_data
    cairo.cairo.cairo_append_path(ctx._pointer, ptr)


_append_paths = (_append_paths_fast if cairo.__name__ == "cairocffi"
                 else _append_paths_slow)


def _append_path(ctx, path, transform, clip=None):
    return _append_paths(ctx, [path], [transform], clip)


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
        'regular'    : cairo.FONT_WEIGHT_NORMAL,
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
        self.dpi = dpi
        self.gc = GraphicsContextCairo(renderer=self)
        self.text_ctx = cairo.Context(
           cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        self.mathtext_parser = MathTextParser('Cairo')
        RendererBase.__init__(self)

    def set_ctx_from_surface(self, surface):
        self.gc.ctx = cairo.Context(surface)
        # Although it may appear natural to automatically call
        # `self.set_width_height(surface.get_width(), surface.get_height())`
        # here (instead of having the caller do so separately), this would fail
        # for PDF/PS/SVG surfaces, which have no way to report their extents.

    def set_width_height(self, width, height):
        self.width  = width
        self.height = height

    def _fill_and_stroke(self, ctx, fill_c, alpha, alpha_overrides):
        if fill_c is not None:
            ctx.save()
            if len(fill_c) == 3 or alpha_overrides:
                ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], alpha)
            else:
                ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], fill_c[3])
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()

    @staticmethod
    @cbook.deprecated("3.0")
    def convert_path(ctx, path, transform, clip=None):
        _append_path(ctx, path, transform, clip)

    def draw_path(self, gc, path, transform, rgbFace=None):
        ctx = gc.ctx
        # Clip the path to the actual rendering extents if it isn't filled.
        clip = (ctx.clip_extents()
                if rgbFace is None and gc.get_hatch() is None
                else None)
        transform = (transform
                     + Affine2D().scale(1, -1).translate(0, self.height))
        ctx.new_path()
        _append_path(ctx, path, transform, clip)
        self._fill_and_stroke(
            ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_markers(self, gc, marker_path, marker_trans, path, transform,
                     rgbFace=None):
        ctx = gc.ctx

        ctx.new_path()
        # Create the path for the marker; it needs to be flipped here already!
        _append_path(ctx, marker_path, marker_trans + Affine2D().scale(1, -1))
        marker_path = ctx.copy_path_flat()

        # Figure out whether the path has a fill
        x1, y1, x2, y2 = ctx.fill_extents()
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            filled = False
            # No fill, just unset this (so we don't try to fill it later on)
            rgbFace = None
        else:
            filled = True

        transform = (transform
                     + Affine2D().scale(1, -1).translate(0, self.height))

        ctx.new_path()
        for i, (vertices, codes) in enumerate(
                path.iter_segments(transform, simplify=False)):
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
                    self._fill_and_stroke(
                        ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

        # Fast path, if there is no fill, draw everything in one step
        if not filled:
            self._fill_and_stroke(
                ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_path_collection(
            self, gc, master_transform, paths, all_transforms, offsets,
            offsetTrans, facecolors, edgecolors, linewidths, linestyles,
            antialiaseds, urls, offset_position):

        path_ids = []
        for path, transform in self._iter_collection_raw_paths(
                master_transform, paths, all_transforms):
            path_ids.append((path, Affine2D(transform)))

        reuse_key = None
        grouped_draw = []

        def _draw_paths():
            if not grouped_draw:
                return
            gc_vars, rgb_fc = reuse_key
            gc = copy.copy(gc0)
            # We actually need to call the setters to reset the internal state.
            vars(gc).update(gc_vars)
            for k, v in gc_vars.items():
                if k == "_linestyle":  # Deprecated, no effect.
                    continue
                try:
                    getattr(gc, "set" + k)(v)
                except (AttributeError, TypeError) as e:
                    pass
            gc.ctx.new_path()
            paths, transforms = zip(*grouped_draw)
            grouped_draw.clear()
            _append_paths(gc.ctx, paths, transforms)
            self._fill_and_stroke(
                gc.ctx, rgb_fc, gc.get_alpha(), gc.get_forced_alpha())

        for xo, yo, path_id, gc0, rgb_fc in self._iter_collection(
                gc, master_transform, all_transforms, path_ids, offsets,
                offsetTrans, facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            path, transform = path_id
            transform = (Affine2D(transform.get_matrix())
                         .translate(xo, yo - self.height).scale(1, -1))
            # rgb_fc could be a ndarray, for which equality is elementwise.
            new_key = vars(gc0), tuple(rgb_fc) if rgb_fc is not None else None
            if new_key == reuse_key:
                grouped_draw.append((path, transform))
            else:
                _draw_paths()
                grouped_draw.append((path, transform))
                reuse_key = new_key
        _draw_paths()

    def draw_image(self, gc, x, y, im):
        im = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(im[::-1])
        surface = cairo.ImageSurface.create_for_data(
            im.ravel().data, cairo.FORMAT_ARGB32,
            im.shape[1], im.shape[0], im.shape[1] * 4)
        ctx = gc.ctx
        y = self.height - y - im.shape[0]

        ctx.save()
        ctx.set_source_surface(surface, float(x), float(y))
        ctx.paint()
        ctx.restore()

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # Note: x,y are device/display coords, not user-coords, unlike other
        # draw_* methods
        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)

        else:
            ctx = gc.ctx
            ctx.new_path()
            ctx.move_to(x, y)
            ctx.select_font_face(prop.get_name(),
                                 self.fontangles[prop.get_style()],
                                 self.fontweights[prop.get_weight()])

            size = prop.get_size_in_points() * self.dpi / 72.0

            ctx.save()
            if angle:
                ctx.rotate(np.deg2rad(-angle))
            ctx.set_font_size(size)

            ctx.show_text(s)
            ctx.restore()

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
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
            ctx.select_font_face(fontProp.name,
                                 self.fontangles[fontProp.style],
                                 self.fontweights[fontProp.weight])

            size = fontsize * self.dpi / 72.0
            ctx.set_font_size(size)
            ctx.show_text(s)

        for ox, oy, w, h in rects:
            ctx.new_path()
            ctx.rectangle(ox, oy, w, h)
            ctx.set_source_rgb(0, 0, 0)
            ctx.fill_preserve()

        ctx.restore()

    def get_canvas_width_height(self):
        return self.width, self.height

    def get_text_width_height_descent(self, s, prop, ismath):
        if ismath:
            width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        ctx = self.text_ctx
        ctx.save()
        ctx.select_font_face(prop.get_name(),
                             self.fontangles[prop.get_style()],
                             self.fontweights[prop.get_weight()])

        # Cairo (says it) uses 1/96 inch user space units, ref: cairo_gstate.c
        # but if /96.0 is used the font is too small
        size = prop.get_size_in_points() * self.dpi / 72

        # problem - scale remembers last setting and font can become
        # enormous causing program to crash
        # save/restore prevents the problem
        ctx.set_font_size(size)

        y_bearing, w, h = ctx.text_extents(s)[1:4]
        ctx.restore()

        return w, h, h + y_bearing

    def new_gc(self):
        self.gc.ctx.save()
        self.gc._alpha = 1
        self.gc._forced_alpha = False # if True, _alpha overrides A from RGBA
        return self.gc

    def points_to_pixels(self, points):
        return points / 72 * self.dpi


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
            self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], _alpha)
        else:
            self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], rgb[3])

    # def set_antialiased(self, b):
        # cairo has many antialiasing modes, we need to pick one for True and
        # one for False.

    def set_capstyle(self, cs):
        if cs in ('butt', 'round', 'projecting'):
            self._capstyle = cs
            self.ctx.set_line_cap(self._capd[cs])
        else:
            raise ValueError('Unrecognized cap style.  Found %s' % cs)

    def set_clip_rectangle(self, rectangle):
        if not rectangle:
            return
        x, y, w, h = np.round(rectangle.bounds)
        ctx = self.ctx
        ctx.new_path()
        ctx.rectangle(x, self.renderer.height - h - y, w, h)
        ctx.clip()

    def set_clip_path(self, path):
        if not path:
            return
        tpath, affine = path.get_transformed_path_and_affine()
        ctx = self.ctx
        ctx.new_path()
        affine = (affine
                  + Affine2D().scale(1, -1).translate(0, self.renderer.height))
        _append_path(ctx, tpath, affine)
        ctx.clip()

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes is None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            self.ctx.set_dash(
                list(self.renderer.points_to_pixels(np.asarray(dashes))),
                offset)

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
        self.ctx.set_line_width(self.renderer.points_to_pixels(w))


class FigureCanvasCairo(FigureCanvasBase):
    supports_blit = False

    def print_png(self, fobj, *args, **kwargs):
        self._get_printed_image_surface().write_to_png(fobj)

    def print_rgba(self, fobj, *args, **kwargs):
        width, height = self.get_width_height()
        buf = self._get_printed_image_surface().get_data()
        fobj.write(cbook._premultiplied_argb32_to_unmultiplied_rgba8888(
            np.asarray(buf).reshape((width, height, 4))))

    print_raw = print_rgba

    def _get_printed_image_surface(self):
        width, height = self.get_width_height()
        renderer = RendererCairo(self.figure.dpi)
        renderer.set_width_height(width, height)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        renderer.set_ctx_from_surface(surface)
        self.figure.draw(renderer)
        return surface

    def print_pdf(self, fobj, *args, **kwargs):
        return self._save(fobj, 'pdf', *args, **kwargs)

    def print_ps(self, fobj, *args, **kwargs):
        return self._save(fobj, 'ps', *args, **kwargs)

    def print_svg(self, fobj, *args, **kwargs):
        return self._save(fobj, 'svg', *args, **kwargs)

    def print_svgz(self, fobj, *args, **kwargs):
        return self._save(fobj, 'svgz', *args, **kwargs)

    def _save(self, fo, fmt, **kwargs):
        # save PDF/PS/SVG
        orientation = kwargs.get('orientation', 'portrait')

        dpi = 72
        self.figure.dpi = dpi
        w_in, h_in = self.figure.get_size_inches()
        width_in_points, height_in_points = w_in * dpi, h_in * dpi

        if orientation == 'landscape':
            width_in_points, height_in_points = (
                height_in_points, width_in_points)

        if fmt == 'ps':
            if not hasattr(cairo, 'PSSurface'):
                raise RuntimeError('cairo has not been compiled with PS '
                                   'support enabled')
            surface = cairo.PSSurface(fo, width_in_points, height_in_points)
        elif fmt == 'pdf':
            if not hasattr(cairo, 'PDFSurface'):
                raise RuntimeError('cairo has not been compiled with PDF '
                                   'support enabled')
            surface = cairo.PDFSurface(fo, width_in_points, height_in_points)
        elif fmt in ('svg', 'svgz'):
            if not hasattr(cairo, 'SVGSurface'):
                raise RuntimeError('cairo has not been compiled with SVG '
                                   'support enabled')
            if fmt == 'svgz':
                if isinstance(fo, str):
                    fo = gzip.GzipFile(fo, 'wb')
                else:
                    fo = gzip.GzipFile(None, 'wb', fileobj=fo)
            surface = cairo.SVGSurface(fo, width_in_points, height_in_points)
        else:
            warnings.warn("unknown format: %s" % fmt, stacklevel=2)
            return

        # surface.set_dpi() can be used
        renderer = RendererCairo(self.figure.dpi)
        renderer.set_width_height(width_in_points, height_in_points)
        renderer.set_ctx_from_surface(surface)
        ctx = renderer.gc.ctx

        if orientation == 'landscape':
            ctx.rotate(np.pi / 2)
            ctx.translate(0, -height_in_points)
            # Perhaps add an '%%Orientation: Landscape' comment?

        self.figure.draw(renderer)

        ctx.show_page()
        surface.finish()
        if fmt == 'svgz':
            fo.close()


@_Backend.export
class _BackendCairo(_Backend):
    FigureCanvas = FigureCanvasCairo
    FigureManager = FigureManagerBase
