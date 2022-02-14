r"""
A module for parsing a subset of the TeX math syntax and rendering it to a
Matplotlib backend.

For a tutorial of its usage, see :doc:`/tutorials/text/mathtext`.  This
document is primarily concerned with implementation details.

The module uses pyparsing_ to parse the TeX expression.

.. _pyparsing: https://pypi.org/project/pyparsing/

The Bakoma distribution of the TeX Computer Modern fonts, and STIX
fonts are supported.  There is experimental support for using
arbitrary fonts, but results may vary without proper tweaking and
metrics for those fonts.
"""

from collections import namedtuple
import functools
import logging

import numpy as np

from matplotlib import _api, rcParams, _mathtext
from matplotlib.ft2font import FT2Image, LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties

_log = logging.getLogger(__name__)


get_unicode_index = _mathtext.get_unicode_index
get_unicode_index.__module__ = __name__


class MathtextBackend:
    """
    The base class for the mathtext backend-specific code.  `MathtextBackend`
    subclasses interface between mathtext and specific Matplotlib graphics
    backends.

    Subclasses need to override the following:

    - :meth:`render_glyph`
    - :meth:`render_rect_filled`
    - :meth:`get_results`

    And optionally, if you need to use a FreeType hinting style:

    - :meth:`get_hinting_type`
    """
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0

    def set_canvas_size(self, w, h, d):
        """Set the dimension of the drawing canvas."""
        self.width  = w
        self.height = h
        self.depth  = d

    def render_glyph(self, ox, oy, info):
        """
        Draw a glyph described by *info* to the reference point (*ox*,
        *oy*).
        """
        raise NotImplementedError()

    def render_rect_filled(self, x1, y1, x2, y2):
        """
        Draw a filled black rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
        raise NotImplementedError()

    def get_results(self, box):
        """
        Return a backend-specific tuple to return to the backend after
        all processing is done.
        """
        raise NotImplementedError()

    def get_hinting_type(self):
        """
        Get the FreeType hinting type to use with this particular
        backend.
        """
        return LOAD_NO_HINTING


class MathtextBackendAgg(MathtextBackend):
    """
    Render glyphs and rectangles to an FTImage buffer, which is later
    transferred to the Agg image by the Agg backend.
    """

    def __init__(self):
        MathtextBackend.__init__(self)
        self._xmin = self._ymin = np.inf
        self._xmax = self._ymax = -np.inf
        self._glyphs = []
        self._rects = []

    ox = cbook.deprecated("3.4")(property(lambda self: 0))
    oy = cbook.deprecated("3.4")(property(lambda self: 0))
    mode = cbook.deprecated("3.4")(property(lambda self: "bbox"))
    image = cbook.deprecated("3.4")(property(lambda self: None))
    bbox = cbook.deprecated("3.4")(property(
        lambda self: [self._xmin, self._ymin, self._xmax, self._ymax]))

    def render_glyph(self, ox, oy, info):
        self._glyphs.append((ox, oy, info))
        metrics = info.metrics
        self._xmin = min(self._xmin, ox + metrics.xmin)
        self._ymin = min(self._ymin, self.height - oy + metrics.ymin)
        self._xmax = max(self._xmax, ox + metrics.xmax)
        self._ymax = max(self._ymax, self.height - oy + metrics.ymax)

    def render_rect_filled(self, x1, y1, x2, y2):
        self._rects.append((x1, y1, x2, y2))
        self._xmin = min(self._xmin, x1)
        self._xmax = max(self._xmax, x2)
        self._ymin = min(self._ymin, y1)
        self._ymax = max(self._ymax, y2)

    def get_results(self, box, used_characters):
        orig_height = box.height
        orig_depth = box.depth
        _mathtext.ship(0, 0, box)
        xmin = np.floor(self._xmin)
        xmax = np.ceil(self._xmax)
        ymin = np.floor(self._ymin)
        ymax = np.ceil(self._ymax)
        dxmin = self._xmin - xmin
        dymin = self._ymin - ymin
        image = FT2Image(np.ceil(xmax - xmin) + 1, np.ceil(ymax - ymin) + 1)

        for ox, oy, info in self._glyphs:
            info.font.draw_glyph_to_bitmap(
                image, ox + dxmin, oy - info.metrics.iceberg + dymin,
                info.glyph, antialiased=rcParams['text.antialiased'])
        for x1, y1, x2, y2 in self._rects:
            x1 += dxmin
            x2 += dymin
            y1 += dxmin
            y2 += dymin
            height = max(int(y2 - y1) - 1, 0)
            if height == 0:
                center = (y2 + y1) / 2
                y = int(center - (height + 1) / 2)
            else:
                y = int(y1)
            image.draw_rect_filled(int(x1), y, np.ceil(x2), y + height)

        return (0, 0,
                np.ceil(xmax - xmin), np.ceil(ymax - ymin), -ymin,
                image,
                used_characters)

    def get_hinting_type(self):
        from matplotlib.backends import backend_agg
        return backend_agg.get_hinting_flag()


class MathtextBackendPath(MathtextBackend):
    """
    Store information to write a mathtext rendering to the text path
    machinery.
    """

    _Result = namedtuple("_Result", "width height depth glyphs rects")

    def __init__(self):
        super().__init__()
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset
        self.glyphs.append((info.font, info.fontsize, info.num, ox, oy))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append((x1, self.height - y2, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._Result(self.width,
                            self.height + self.depth,
                            self.depth,
                            self.glyphs,
                            self.rects)


class MathTextWarning(Warning):
    pass


##############################################################################
# MAIN


class MathTextParser:
    _parser = None

    _backend_mapping = {
        'bitmap': MathtextBackendPath,
        'agg':    MathtextBackendAgg,
        'ps':     MathtextBackendPath,
        'pdf':    MathtextBackendPath,
        'svg':    MathtextBackendPath,
        'path':   MathtextBackendPath,
        'cairo':  MathtextBackendPath,
        'macosx': MathtextBackendAgg,
    }
    _font_type_mapping = {
        'cm':          _mathtext.BakomaFonts,
        'dejavuserif': _mathtext.DejaVuSerifFonts,
        'dejavusans':  _mathtext.DejaVuSansFonts,
        'stix':        _mathtext.StixFonts,
        'stixsans':    _mathtext.StixSansFonts,
        'custom':      _mathtext.UnicodeFonts,
    }

    def __init__(self, output):
        """Create a MathTextParser for the given backend *output*."""
        self._output = output.lower()

    def parse(self, s, dpi=72, prop=None):
        """
        Parse the given math expression *s* at the given *dpi*.  If *prop* is
        provided, it is a `.FontProperties` object specifying the "default"
        font to use in the math expression, used for all non-math text.

        The results are cached, so multiple calls to `parse`
        with the same expression should be fast.
        """
        # lru_cache can't decorate parse() directly because the ps.useafm and
        # mathtext.fontset rcParams also affect the parse (e.g. by affecting
        # the glyph metrics).
        return self._parse_cached(s, dpi, prop)

    @functools.lru_cache(50)
    def _parse_cached(self, s, dpi, prop):
        if prop is None:
            prop = FontProperties()

        fontset_class = _api.check_getitem(
                self._font_type_mapping, fontset=prop.get_math_fontfamily())
        backend = self._backend_mapping[self._output]()
        font_output = fontset_class(prop, backend)

        fontsize = prop.get_size_in_points()

        # This is a class variable so we don't rebuild the parser
        # with each request.
        if self._parser is None:
            self.__class__._parser = _mathtext.Parser()

        box = self._parser.parse(s, font_output, fontsize, dpi)
        font_output.set_canvas_size(box.width, box.height, box.depth)
        return font_output.get_results(box)


def math_to_image(s, filename_or_obj, prop=None, dpi=None, format=None,
                  *, color=None):
    """
    Given a math expression, renders it in a closely-clipped bounding
    box to an image file.

    Parameters
    ----------
    s : str
        A math expression.  The math portion must be enclosed in dollar signs.
    filename_or_obj : str or path-like or file-like
        Where to write the image data.
    prop : `.FontProperties`, optional
        The size and style of the text.
    dpi : float, optional
        The output dpi.  If not set, the dpi is determined as for
        `.Figure.savefig`.
    format : str, optional
        The output format, e.g., 'svg', 'pdf', 'ps' or 'png'.  If not set, the
        format is determined as for `.Figure.savefig`.
    color : str, optional
        Foreground color, defaults to :rc:`text.color`.
    """
    from matplotlib import figure

    parser = MathTextParser('path')
    width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)

    fig = figure.Figure(figsize=(width / 72.0, height / 72.0))
    fig.text(0, depth/height, s, fontproperties=prop, color=color)
    fig.savefig(filename_or_obj, dpi=dpi, format=format)

    return depth
