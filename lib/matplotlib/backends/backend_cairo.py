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

from __future__ import division
import os, sys, warnings, gzip

import numpy as npy

def _fn_name(): return sys._getframe(1).f_code.co_name

import cairo
_version_required = (1,2,0)
if cairo.version_info < _version_required:
   raise SystemExit ("Pycairo %d.%d.%d is installed\n"
                     "Pycairo %d.%d.%d or later is required"
                     % (cairo.version_info + _version_required))
backend_version = cairo.version
del _version_required

from matplotlib import agg
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook        import enumerate, izip, is_string_like
from matplotlib.figure       import Figure
from matplotlib.mathtext     import MathTextParser
from matplotlib.transforms   import Bbox
from matplotlib.font_manager import ttfFontProperty
from matplotlib import rcParams

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
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        self.dpi = dpi
        self.text_ctx = cairo.Context (
           cairo.ImageSurface (cairo.FORMAT_ARGB32,1,1))
        self.mathtext_parser = MathTextParser('Cairo')

    def set_ctx_from_surface (self, surface):
       self.ctx = cairo.Context (surface)
       self.ctx.save() # restore, save  - when call new_gc()


    def set_width_height(self, width, height):
        self.width  = width
        self.height = height
        self.matrix_flipy = cairo.Matrix (yy=-1, y0=self.height)
        # use matrix_flipy for ALL rendering?
        # - problem with text? - will need to switch matrix_flipy off, or do a
        # font transform?


    def _fill_and_stroke (self, ctx, fill_c):
        #assert fill_c or stroke_c

        #_.ctx.save()

        if fill_c:
            ctx.save()
            ctx.set_source_rgb (*fill_c)
            #if stroke_c:   # always (implicitly) set at the moment
            ctx.fill_preserve()
            #else:
            #    ctx.fill()
            ctx.restore()

        #if stroke_c:                      # always stroke
            #ctx.set_source_rgb (stroke_c) # is already set
        ctx.stroke()

        #_.ctx.restore() # revert to the default attributes

    def draw_path(self, gc, rgbFace, path):
        ctx = gc.ctx
        ctx.new_path()

        while 1:
            code, xp, yp = path.vertex()
            yp = self.height - yp

            if code == agg.path_cmd_stop:
                ctx.close_path()
                break
            elif code == agg.path_cmd_move_to:
                ctx.move_to(xp, yp)
            elif code == agg.path_cmd_line_to:
                ctx.line_to(xp, yp)
            elif code == agg.path_cmd_curve3:
                _, xp1, yp1 = path.vertex()
                yp1 = self.height - yp1
                ctx.curve_to(xp, yp, xp, yp, xp1, yp1)
            elif code == agg.path_cmd_curve4:
                _, xp1, yp1 = path.vertex()
                yp1 = self.height - yp1
                _, xp2, yp2 = path.vertex()
                yp2 = self.height - yp2
                ctx.curve_to(xp, yp, xp1, yp1, xp2, yp2)
            elif code == agg.path_cmd_end_poly:
                ctx.close_path()
        self._fill_and_stroke(ctx, rgbFace)

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2,
                 rotation):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        ctx = gc.ctx
        ctx.save()
        ctx.translate(x, self.height - y)
        ctx.rotate(rotation)
        ctx.scale(width / 2.0, height / 2.0)
        ctx.new_sub_path()
        ctx.arc(0.0, 0.0, 1.0, npy.pi * angle1 / 180.,
                npy.pi * angle2 / 180.)
        ctx.restore()

        self._fill_and_stroke (ctx, rgbFace)


    def draw_image(self, x, y, im, bbox):
        # bbox - not currently used
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        im.flipud_out()

        rows, cols, buf = im.color_conv (BYTE_FORMAT)
        surface = cairo.ImageSurface.create_for_data (
                      buf, cairo.FORMAT_ARGB32, cols, rows, cols*4)
        # function does not pass a 'gc' so use renderer.ctx
        ctx = self.ctx
        y = self.height - y - rows
        ctx.set_source_surface (surface, x, y)
        ctx.paint()

        im.flipud_out()


    def draw_line(self, gc, x1, y1, x2, y2):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        ctx = gc.ctx
        ctx.new_path()
        ctx.move_to (x1, self.height - y1)
        ctx.line_to (x2, self.height - y2)
        self._fill_and_stroke (ctx, None)


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
        self._fill_and_stroke (ctx, None)

        ctx.set_matrix (matrix_old)


    def draw_markers_OLD(self, gc, path, rgbFace, x, y, transform):
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

            self._fill_and_stroke (ctx, rgbFace)

            ctx.restore() # undo translate()

        #ctx.set_matrix(matrix_old)


    def draw_point(self, gc, x, y):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        # render by drawing a 0.5 radius circle
        ctx = gc.ctx
        ctx.new_path()
        ctx.arc (x, self.height - y, 0.5, 0, 2*npy.pi)
        self._fill_and_stroke (ctx, gc.get_rgb())


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

        self._fill_and_stroke (ctx, rgbFace)

        ctx.set_matrix (matrix_old)

    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        ctx = gc.ctx
        ctx.new_path()
        ctx.rectangle (x, self.height - y - height, width, height)
        self._fill_and_stroke (ctx, rgbFace)


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
              ctx.rotate (-angle * npy.pi / 180)
           ctx.set_font_size (size)
           ctx.show_text (s.encode("utf-8"))
           ctx.restore()

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())

        ctx = gc.ctx
        width, height, descent, glyphs, rects = self.mathtext_parser.parse(
            s, self.dpi.get(), prop)

        ctx.save()
        ctx.translate(x, y)
        if angle:
           ctx.rotate (-angle * npy.pi / 180)

        for font, fontsize, s, ox, oy in glyphs:
           ctx.new_path()
           ctx.move_to(ox, oy)

           fontProp = ttfFontProperty(font)
           ctx.save()
           ctx.select_font_face (fontProp.name,
                                 self.fontangles [fontProp.style],
                                 self.fontweights[fontProp.weight])

           # size = prop.get_size_in_points() * self.dpi.get() / 96.0
           size = fontsize * self.dpi.get() / 72.0
           ctx.set_font_size(size)
           ctx.show_text(s.encode("utf-8"))
           ctx.restore()

        for ox, oy, w, h in rects:
           ctx.new_path()
           ctx.rectangle (ox, oy, w, h)
           ctx.set_source_rgb (0, 0, 0)
           ctx.fill_preserve()

        ctx.restore()


    def flipy(self):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        return True
        #return False # tried - all draw objects ok except text (and images?)
        # which comes out mirrored!


    def get_canvas_width_height(self):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        return self.width, self.height


    def get_text_width_height_descent(self, s, prop, ismath):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        if ismath:
            width, height, descent, fonts, used_characters = self.mathtext_parser.parse(
               s, self.dpi.get(), prop)
            return width, height, descent

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

        y_bearing, w, h = ctx.text_extents (s)[1:4]
        ctx.restore()

        return w, h, h + y_bearing


    def new_gc(self):
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        self.ctx.restore()  # matches save() in set_ctx_from_surface()
        self.ctx.save()
        return GraphicsContextCairo (renderer=self)


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


    def __init__(self, renderer):
        GraphicsContextBase.__init__(self)
        self.renderer = renderer
        self.ctx = renderer.ctx


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
        self._cliprect = rectangle

        x,y,w,h = rectangle
        # pixel-aligned clip-regions are faster
        x,y,w,h = round(x), round(y), round(w), round(h)
        ctx = self.ctx
        ctx.new_path()
        ctx.rectangle (x, self.renderer.height - h - y, w, h)
        ctx.clip ()
        # Alternative: just set _cliprect here and actually set cairo clip rect
        # in fill_and_stroke() inside ctx.save() ... ctx.restore()


    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes == None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            self.ctx.set_dash (
               self.renderer.points_to_pixels (npy.asarray(dashes)), offset)


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
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas  = FigureCanvasCairo(thisFig)
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

    def get_default_filetype(self):
        return rcParams['cairo.format']

    def _save (self, fo, format, **kwargs):
        # save PDF/PS/SVG
        orientation = kwargs.get('orientation', 'portrait')

        dpi = 72
        self.figure.dpi.set (dpi)
        w_in, h_in = self.figure.get_size_inches()
        width_in_points, height_in_points = w_in * dpi, h_in * dpi

        if orientation == 'landscape':
            width_in_points, height_in_points = (height_in_points,
                                                 width_in_points)

        if format == 'ps':
            if not cairo.HAS_PS_SURFACE:
                raise RuntimeError ('cairo has not been compiled with PS '
                                    'support enabled')
            surface = cairo.PSSurface (fo, width_in_points, height_in_points)
        elif format == 'pdf':
            if not cairo.HAS_PDF_SURFACE:
                raise RuntimeError ('cairo has not been compiled with PDF '
                                    'support enabled')
            surface = cairo.PDFSurface (fo, width_in_points, height_in_points)
        elif format in ('svg', 'svgz'):
            if not cairo.HAS_SVG_SURFACE:
                raise RuntimeError ('cairo has not been compiled with SVG '
                                    'support enabled')
            if format == 'svgz':
                filename = fo
                if is_string_like(fo):
                    fo = open(fo, 'wb')
                fo = gzip.GzipFile(None, 'wb', fileobj=fo)
            surface = cairo.SVGSurface (fo, width_in_points, height_in_points)
        else:
           warnings.warn ("unknown format: %s" % format)
           return

        # surface.set_dpi() can be used
        renderer = RendererCairo (self.figure.dpi)
        renderer.set_width_height (width_in_points, height_in_points)
        renderer.set_ctx_from_surface (surface)
        ctx = renderer.ctx

        if orientation == 'landscape':
            ctx.rotate (npy.pi/2)
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
