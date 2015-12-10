"""
An agg http://antigrain.com/ backend

Features that are implemented

 * capstyles and join styles
 * dashes
 * linewidth
 * lines, rectangles, ellipses
 * clipping to a rectangle
 * output to RGBA and PNG, optionally JPEG and TIFF
 * alpha blending
 * DPI scaling properly - everything scales properly (dashes, linewidths, etc)
 * draw polygon
 * freetype2 w/ ft2font

TODO:

  * integrate screen dpi w/ ppi and text

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import threading
import numpy as np
from math import radians, cos, sin
from matplotlib import verbose, rcParams
from matplotlib.backend_bases import (RendererBase, FigureManagerBase,
                                      FigureCanvasBase)
from matplotlib.cbook import is_string_like, maxdict, restrict_dict
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
                                LOAD_DEFAULT, LOAD_NO_AUTOHINT)
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib import colors as mcolors

from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
from matplotlib import _png

try:
    from PIL import Image
    _has_pil = True
except ImportError:
    _has_pil = False

backend_version = 'v2.2'

def get_hinting_flag():
    mapping = {
        True: LOAD_FORCE_AUTOHINT,
        False: LOAD_NO_HINTING,
        'either': LOAD_DEFAULT,
        'native': LOAD_NO_AUTOHINT,
        'auto': LOAD_FORCE_AUTOHINT,
        'none': LOAD_NO_HINTING
        }
    return mapping[rcParams['text.hinting']]


class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """
    debug=1

    # we want to cache the fonts at the class level so that when
    # multiple figures are created we can reuse them.  This helps with
    # a bug on windows where the creation of too many figures leads to
    # too many open file handles.  However, storing them at the class
    # level is not thread safe.  The solution here is to let the
    # FigureCanvas acquire a lock on the fontd at the start of the
    # draw, and release it when it is done.  This allows multiple
    # renderers to share the cached fonts, but only one figure can
    # draw at at time and so the font cache is used by only one
    # renderer at a time

    lock = threading.RLock()
    def __init__(self, width, height, dpi):
        if __debug__: verbose.report('RendererAgg.__init__', 'debug-annoying')
        RendererBase.__init__(self)

        self.dpi = dpi
        self.width = width
        self.height = height
        if __debug__: verbose.report('RendererAgg.__init__ width=%s, height=%s'%(width, height), 'debug-annoying')
        self._renderer = _RendererAgg(int(width), int(height), dpi, debug=False)
        self._filter_renderers = []

        if __debug__: verbose.report('RendererAgg.__init__ _RendererAgg done',
                                     'debug-annoying')

        self._update_methods()
        self.mathtext_parser = MathTextParser('Agg')

        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)
        if __debug__: verbose.report('RendererAgg.__init__ done',
                                     'debug-annoying')

    def __getstate__(self):
        # We only want to preserve the init keywords of the Renderer.
        # Anything else can be re-created.
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        self.__init__(state['width'], state['height'], state['dpi'])

    def _get_hinting_flag(self):
        if rcParams['text.hinting']:
            return LOAD_FORCE_AUTOHINT
        else:
            return LOAD_NO_HINTING

    # for filtering to work with rasterization, methods needs to be wrapped.
    # maybe there is better way to do it.
    def draw_markers(self, *kl, **kw):
        return self._renderer.draw_markers(*kl, **kw)

    def draw_path_collection(self, *kl, **kw):
        return self._renderer.draw_path_collection(*kl, **kw)

    def _update_methods(self):
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.draw_gouraud_triangle = self._renderer.draw_gouraud_triangle
        self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
        self.draw_image = self._renderer.draw_image
        self.copy_from_bbox = self._renderer.copy_from_bbox
        self.get_content_extents = self._renderer.get_content_extents

    def tostring_rgba_minimized(self):
        extents = self.get_content_extents()
        bbox = [[extents[0], self.height - (extents[1] + extents[3])],
                [extents[0] + extents[2], self.height - extents[1]]]
        region = self.copy_from_bbox(bbox)
        return np.array(region), extents

    def draw_path(self, gc, path, transform, rgbFace=None):
        """
        Draw the path
        """
        nmax = rcParams['agg.path.chunksize'] # here at least for testing
        npts = path.vertices.shape[0]
        if (nmax > 100 and npts > nmax and path.should_simplify and
                rgbFace is None and gc.get_hatch() is None):
            nch = np.ceil(npts/float(nmax))
            chsize = int(np.ceil(npts/nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
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

        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))
        x = np.round(x + ox + xd)
        y = np.round(y - oy + yd)
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """
        Render the text
        """
        if __debug__: verbose.report('RendererAgg.draw_text', 'debug-annoying')

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        flags = get_hinting_flag()
        font = self._get_agg_font(prop)

        if font is None: return None
        if len(s) == 1 and ord(s) > 127:
            font.load_char(ord(s), flags=flags)
        else:
            # We pass '0' for angle here, since it will be rotated (in raster
            # space) in the following call to draw_text_image).
            font.set_text(s, 0, flags=flags)
        font.draw_glyphs_to_bitmap(antialiased=rcParams['text.antialiased'])
        d = font.get_descent() / 64.0
        # The descent needs to be adjusted for the angle
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = -d * sin(radians(angle))
        yd = d * cos(radians(angle))

        #print x, y, int(x), int(y), s
        self._renderer.draw_text_image(
            font, round(x - xd + xo), round(y + yd + yo) + 1, angle, gc)

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop

        # passing rgb is a little hack to make caching in the
        # texmanager more efficient.  It is not meant to be used
        # outside the backend
        """
        if rcParams['text.usetex']:
            # todo: handle props
            size = prop.get_size_in_points()
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            w, h, d = texmanager.get_text_width_height_descent(s, fontsize,
                                                               renderer=self)
            return w, h, d

        if ismath:
            ox, oy, width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        flags = get_hinting_flag()
        font = self._get_agg_font(prop)
        font.set_text(s, 0.0, flags=flags)  # the width and height of unrotated string
        w, h = font.get_width_height()
        d = font.get_descent()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d /= 64.0
        return w, h, d

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!', mtext=None):
        # todo, handle props, angle, origins
        size = prop.get_size_in_points()

        texmanager = self.get_texmanager()

        Z = texmanager.get_grey(s, size, self.dpi)
        Z = np.array(Z * 255.0, np.uint8)

        w, h, d = self.get_text_width_height_descent(s, prop, ismath)
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xd)
        y = round(y + yd)

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

        fname = findfont(prop)
        font = get_font(
            fname,
            hinting_factor=rcParams['text.hinting_factor'])

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

    def buffer_rgba(self):
        if __debug__: verbose.report('RendererAgg.buffer_rgba',
                                     'debug-annoying')
        return self._renderer.buffer_rgba()

    def clear(self):
        self._renderer.clear()

    def option_image_nocomposite(self):
        # It is generally faster to composite each image directly to
        # the Figure, and there's no file size benefit to compositing
        # with the Agg backend
        return True

    def option_scale_image(self):
        """
        agg backend support arbitrary scaling of image.
        """
        return True

    def restore_region(self, region, bbox=None, xy=None):
        """
        Restore the saved region. If bbox (instance of BboxBase, or
        its extents) is given, only the region specified by the bbox
        will be restored. *xy* (a tuple of two floasts) optionally
        specifies the new position (the LLC of the original region,
        not the LLC of the bbox) where the region will be restored.

        >>> region = renderer.copy_from_bbox()
        >>> x1, y1, x2, y2 = region.get_extents()
        >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),
        ...                         xy=(x1-dx, y1))

        """
        if bbox is not None or xy is not None:
            if bbox is None:
                x1, y1, x2, y2 = region.get_extents()
            elif isinstance(bbox, BboxBase):
                x1, y1, x2, y2 = bbox.extents
            else:
                x1, y1, x2, y2 = bbox

            if xy is None:
                ox, oy = x1, y1
            else:
                ox, oy = xy

            # The incoming data is float, but the _renderer type-checking wants
            # to see integers.
            self._renderer.restore_region(region, int(x1), int(y1),
                                          int(x2), int(y2), int(ox), int(oy))

        else:
            self._renderer.restore_region(region)

    def start_filter(self):
        """
        Start filtering. It simply create a new canvas (the old one is saved).
        """
        self._filter_renderers.append(self._renderer)
        self._renderer = _RendererAgg(int(self.width), int(self.height),
                                      self.dpi)
        self._update_methods()

    def stop_filter(self, post_processing):
        """
        Save the plot in the current canvas as a image and apply
        the *post_processing* function.

           def post_processing(image, dpi):
             # ny, nx, depth = image.shape
             # image (numpy array) has RGBA channels and has a depth of 4.
             ...
             # create a new_image (numpy array of 4 channels, size can be
             # different). The resulting image may have offsets from
             # lower-left corner of the original image
             return new_image, offset_x, offset_y

        The saved renderer is restored and the returned image from
        post_processing is plotted (using draw_image) on it.
        """

        # WARNING.
        # For agg_filter to work, the rendere's method need
        # to overridden in the class. See draw_markers, and draw_path_collections

        from matplotlib._image import fromarray

        width, height = int(self.width), int(self.height)

        buffer, bounds = self.tostring_rgba_minimized()

        l, b, w, h = bounds


        self._renderer = self._filter_renderers.pop()
        self._update_methods()

        if w > 0 and h > 0:
            img = np.fromstring(buffer, np.uint8)
            img, ox, oy = post_processing(img.reshape((h, w, 4)) / 255.,
                                          self.dpi)
            image = fromarray(img, 1)

            gc = self.new_gc()
            self._renderer.draw_image(gc,
                                      l+ox, height - b - h +oy,
                                      image)


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if __debug__: verbose.report('backend_agg.new_figure_manager',
                                 'debug-annoying')


    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasAgg(figure)
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

    def restore_region(self, region, bbox=None, xy=None):
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        """
        Draw the figure using the renderer
        """
        if __debug__: verbose.report('FigureCanvasAgg.draw', 'debug-annoying')

        self.renderer = self.get_renderer(cleared=True)
        # acquire a lock on the shared font cache
        RendererAgg.lock.acquire()

        try:
            self.figure.draw(self.renderer)
        finally:
            RendererAgg.lock.release()

    def get_renderer(self, cleared=False):
        l, b, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi
        try: self._lastKey, self.renderer
        except AttributeError: need_new_renderer = True
        else:  need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        elif cleared:
            self.renderer.clear()
        return self.renderer

    def tostring_rgb(self):
        if __debug__: verbose.report('FigureCanvasAgg.tostring_rgb',
                                     'debug-annoying')
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        if __debug__: verbose.report('FigureCanvasAgg.tostring_argb',
                                     'debug-annoying')
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        if __debug__: verbose.report('FigureCanvasAgg.buffer_rgba',
                                     'debug-annoying')
        return self.renderer.buffer_rgba()

    def print_raw(self, filename_or_obj, *args, **kwargs):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = self.figure.dpi
        if is_string_like(filename_or_obj):
            fileobj = open(filename_or_obj, 'wb')
            close = True
        else:
            fileobj = filename_or_obj
            close = False
        try:
            fileobj.write(renderer._renderer.buffer_rgba())
        finally:
            if close:
                filename_or_obj.close()
        renderer.dpi = original_dpi
    print_rgba = print_raw

    def print_png(self, filename_or_obj, *args, **kwargs):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = self.figure.dpi
        if is_string_like(filename_or_obj):
            filename_or_obj = open(filename_or_obj, 'wb')
            close = True
        else:
            close = False

        try:
            _png.write_png(renderer._renderer, filename_or_obj, self.figure.dpi)
        finally:
            if close:
                filename_or_obj.close()
        renderer.dpi = original_dpi

    def print_to_buffer(self):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = self.figure.dpi
        result = (renderer._renderer.buffer_rgba(),
                  (int(renderer.width), int(renderer.height)))
        renderer.dpi = original_dpi
        return result

    if _has_pil:
        # add JPEG support
        def print_jpg(self, filename_or_obj, *args, **kwargs):
            """
            Supported kwargs:

            *quality*: The image quality, on a scale from 1 (worst) to
                95 (best). The default is 95, if not given in the
                matplotlibrc file in the savefig.jpeg_quality parameter.
                Values above 95 should be avoided; 100 completely
                disables the JPEG quantization stage.

            *optimize*: If present, indicates that the encoder should
                make an extra pass over the image in order to select
                optimal encoder settings.

            *progressive*: If present, indicates that this image
                should be stored as a progressive JPEG file.
            """
            buf, size = self.print_to_buffer()
            if kwargs.pop("dryrun", False):
                return
            # The image is "pasted" onto a white background image to safely
            # handle any transparency
            image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
            color = mcolors.colorConverter.to_rgb(
                rcParams.get('savefig.facecolor', 'white'))
            color = tuple([int(x * 255.0) for x in color])
            background = Image.new('RGB', size, color)
            background.paste(image, image)
            options = restrict_dict(kwargs, ['quality', 'optimize',
                                             'progressive'])

            if 'quality' not in options:
                options['quality'] = rcParams['savefig.jpeg_quality']

            return background.save(filename_or_obj, format='jpeg', **options)
        print_jpeg = print_jpg

        # add TIFF support
        def print_tif(self, filename_or_obj, *args, **kwargs):
            buf, size = self.print_to_buffer()
            if kwargs.pop("dryrun", False):
                return
            image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
            dpi = (self.figure.dpi, self.figure.dpi)
            return image.save(filename_or_obj, format='tiff',
                              dpi=dpi)
        print_tiff = print_tif


FigureCanvas = FigureCanvasAgg
