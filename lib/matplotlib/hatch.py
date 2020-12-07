"""Contains classes for generating hatch patterns."""
from collections.abc import Iterable

import numpy as np

from matplotlib import _api, docstring
from matplotlib.path import Path


class HatchPatternBase:
    """The base class for a hatch pattern."""
    pass


class HorizontalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('-') + hatch.count('+')) * density)
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = 0.0
        vertices[0::2, 1] = steps
        vertices[1::2, 0] = 1.0
        vertices[1::2, 1] = steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class VerticalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('|') + hatch.count('+')) * density)
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = steps
        vertices[0::2, 1] = 0.0
        vertices[1::2, 0] = steps
        vertices[1::2, 1] = 1.0
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class NorthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int(
            (hatch.count('/') + hatch.count('x') + hatch.count('X')) * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 0.0 - steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 1.0 - steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class SouthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int(
            (hatch.count('\\') + hatch.count('x') + hatch.count('X'))
            * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 1.0 + steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 0.0 + steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class Shapes(HatchPatternBase):
    filled = False

    def __init__(self, hatch, density):
        if self.num_rows == 0:
            self.num_shapes = 0
            self.num_vertices = 0
        else:
            self.num_shapes = ((self.num_rows // 2 + 1) * (self.num_rows + 1) +
                               (self.num_rows // 2) * self.num_rows)
            self.num_vertices = (self.num_shapes *
                                 len(self.shape_vertices) *
                                 (1 if self.filled else 2))

    def set_vertices_and_codes(self, vertices, codes):
        offset = 1.0 / self.num_rows
        shape_vertices = self.shape_vertices * offset * self.size
        if not self.filled:
            inner_vertices = shape_vertices[::-1] * 0.9
        shape_codes = self.shape_codes
        shape_size = len(shape_vertices)

        cursor = 0
        for row in range(self.num_rows + 1):
            if row % 2 == 0:
                cols = np.linspace(0, 1, self.num_rows + 1)
            else:
                cols = np.linspace(offset / 2, 1 - offset / 2, self.num_rows)
            row_pos = row * offset
            for col_pos in cols:
                vertices[cursor:cursor + shape_size] = (shape_vertices +
                                                        (col_pos, row_pos))
                codes[cursor:cursor + shape_size] = shape_codes
                cursor += shape_size
                if not self.filled:
                    vertices[cursor:cursor + shape_size] = (inner_vertices +
                                                            (col_pos, row_pos))
                    codes[cursor:cursor + shape_size] = shape_codes
                    cursor += shape_size


class Circles(Shapes):
    def __init__(self, hatch, density):
        path = Path.unit_circle()
        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        super().__init__(hatch, density)


class SmallCircles(Circles):
    size = 0.2

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('o')) * density
        super().__init__(hatch, density)


class LargeCircles(Circles):
    size = 0.35

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('O')) * density
        super().__init__(hatch, density)


# TODO: __init__ and class attributes override all attributes set by
# SmallCircles. Should this class derive from Circles instead?
class SmallFilledCircles(SmallCircles):
    size = 0.1
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('.')) * density
        # Not super().__init__!
        Circles.__init__(self, hatch, density)


class Stars(Shapes):
    size = 1.0 / 3.0
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('*')) * density
        path = Path.unit_regular_star(5)
        self.shape_vertices = path.vertices
        self.shape_codes = np.full(len(self.shape_vertices), Path.LINETO,
                                   dtype=Path.code_type)
        self.shape_codes[0] = Path.MOVETO
        super().__init__(hatch, density)

_hatch_types = [
    HorizontalHatch,
    VerticalHatch,
    NorthEastHatch,
    SouthEastHatch,
    SmallCircles,
    LargeCircles,
    SmallFilledCircles,
    Stars
    ]


class Hatch:
    r"""
    Pattern to be tiled within a filled area.

    For a visual example of the available *Hatch* patterns, `view these docs
    online <Hatch>` or run `Hatch.demo`.

    When making plots that contain multiple filled shapes, like :doc:`bar
    charts </gallery/lines_bars_and_markers/bar_stacked>` or filled
    :doc:`countour plots </images_contours_and_fields/contourf_hatching>`, it
    is common to use :doc:`color </tutorials/colors/colors>` to distinguish
    between areas. However, if color is not available, such as when printing in
    black and white, Matplotlib also supports hatching (i.e. filling each
    area with a unique repeating pattern or lines or shapes) in order to make
    it easy to refer to a specific filled bar, shape, or similar.

    .. warning::
        Hatching is currently only supported in the Agg, PostScript, PDF, and
        SVG backends.

    **Hatching patterns**

    There hatching primitives built into Matplotlib are:

    .. rst-class:: value-list

        '-'
            Horizontal lines.
        '|'
            Vertical lines.
        '+'
            Crossed lines. ``'+'`` is equivalent to ``'-|'``.
        '\'
            Diagonal lines running northwest to southeast.
        '/'
            Diagonal lines running southwest to northeast.
        'x'
            Crossed diagonal lines. Equivalent to ``r'\/'``.
        'X'
            Synonym for ``'x'``.
        '.'
            Dots (i.e. very small, filled circles).
        'o'
            Small, unfilled circles.
        'O'
            Large, unfilled circles.
        '*'
            Filled star shape.

    Hatching primitives can be combined to make more complicated patterns. For
    example, a hatch pattern of ``'*/|'`` would fill the area with vertical and
    diagonal lines as well as stars.

    **Hatching Density**

    By default, the hatching pattern is tiled so that there are **6** lines per
    inch (in display space), but this can be tuned (in integer increments)
    using the *density* kwarg to *Hatch*.

    For convenience, the same symbol can also be repeated to request a higher
    hatching density. For example, ``'||-'`` will have twice as many vertical
    lines as ``'|-'``.  Notice that since ``'|-'`` can also be written as
    ``'+'``, we can also write ``'||-'`` as ``'|+'``.

    Examples
    --------
    For more examples of how to use hatching, see `the hatching demo
    </gallery/shapes_and_collections/hatch_demo>` and `the contourf hatching
    demo </gallery/images_contours_and_fields/contourf_hatching>`.

    .. plot::
        :alt: Demo showing each hatching primitive at its default density.

        from matplotlib.hatch import Hatch
        Hatch.demo()
    """

    _default_density = 6
    _valid_hatch_patterns = set(r'-|+/\xX.oO*')

    def __init__(self, pattern_spec, density=None):
        self.density = Hatch._default_density if density is None else density
        self._pattern_spec = pattern_spec
        self.patterns = self._validate_hatch_pattern(pattern_spec)
        self._build_path()

    @classmethod
    def from_path(cls, path):
        hatch = cls(None, 0)
        hatch.path = path

    def _build_path(self):
        # the API of HatchPatternBase was architected before Hatch, so instead
        # of simply returning Path's that we can concatenate using
        # Path.make_compound_path, we must pre-allocate the vertices array for
        # the final path up front. (The performance gain from this
        # preallocation is untested).
        num_vertices = sum([pattern.num_vertices for pattern in self.patterns])

        if num_vertices == 0:
            self.path = Path(np.empty((0, 2)))

        vertices = np.empty((num_vertices, 2))
        codes = np.empty(num_vertices, Path.code_type)

        cursor = 0
        for pattern in self.patterns:
            if pattern.num_vertices != 0:
                vertices_chunk = vertices[cursor:cursor + pattern.num_vertices]
                codes_chunk = codes[cursor:cursor + pattern.num_vertices]
                pattern.set_vertices_and_codes(vertices_chunk, codes_chunk)
                cursor += pattern.num_vertices

        self.path = Path(vertices, codes)

    def _validate_hatch_pattern(self, patterns):
        if isinstance(patterns, Hatch):
            patterns = patterns._pattern_spec
        if patterns is None or patterns is []:
            return []
        elif isinstance(patterns, str):
            invalids = set(patterns).difference(Hatch._valid_hatch_patterns)
            if invalids:
                Hatch._warn_invalid_hatch(invalids)
            return [hatch_type(patterns, self.density)
                    for hatch_type in _hatch_types]
        elif isinstance(patterns, Iterable) and np.all([
                isinstance(p, HatchPatternBase) for p in patterns]):
            return patterns
        else:
            raise ValueError(f"Cannot construct hatch pattern from {patterns}")

    def _warn_invalid_hatch(invalids):
        valid = ''.join(sorted(Hatch._valid_hatch_patterns))
        invalids = ''.join(sorted(invalids))
        _api.warn_deprecated(
            '3.4',
            message=f'hatch must consist of a string of "{valid}" or None, '
                    f'but found the following invalid values "{invalids}". '
                    'Passing invalid values is deprecated since %(since)s and '
                    'will become an error %(removal)s.'
        )

    @staticmethod
    def demo(density=6):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        num_patches = len(Hatch._valid_hatch_patterns)

        spacing = 0.1  # percent of width
        boxes_per_row = 4
        num_rows = np.ceil(num_patches / boxes_per_row)
        inter_box_dist_y = 1/num_rows
        posts = np.linspace(0, 1, boxes_per_row + 1)
        inter_box_dist_x = posts[1] - posts[0]
        font_size = 12
        fig_size = (4, 4)
        text_pad = 0.2  # fraction of text height
        text_height = (1 + text_pad)*(
            fig.dpi_scale_trans + ax.transAxes.inverted()
        ).transform([1, 1])[1]
        # half of text pad
        bottom_padding = text_height*(1 - (1/(1+text_pad)))/2

        for i, hatch in enumerate(Hatch._valid_hatch_patterns):
            row = int(i/boxes_per_row)
            col = i - row*boxes_per_row
            ax.add_patch(Rectangle(
                xy=[(col + spacing/2) * inter_box_dist_x,
                    bottom_padding + row*inter_box_dist_y],
                width=inter_box_dist_x*(1 - spacing),
                height=inter_box_dist_y*(1 - text_height),
                transform=ax.transAxes,
                hatch=hatch,
                label="'" + hatch + "'"
            ))
            ax.text((col + 1/2) * inter_box_dist_x,
                    bottom_padding + (-text_height*(1/(1+text_pad)) + row
                    + 1)*inter_box_dist_y,
                    "'" + hatch + "'", horizontalalignment='center',
                    fontsize=font_size)


Hatch.input_description = "{" \
        + ", ".join([f"'{p}'" for p in Hatch._valid_hatch_patterns]) \
        + "}"


docstring.interpd.update({'Hatch': Hatch.input_description})


@_api.deprecated("3.4")
def get_path(hatchpattern, density=6):
    """
    Given a hatch specifier, *hatchpattern*, generates Path to render
    the hatch in a unit square.  *density* is the number of lines per
    unit square.
    """
    return Hatch(hatchpattern).path

docstring
