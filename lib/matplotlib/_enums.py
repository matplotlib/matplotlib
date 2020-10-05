"""
Enums representing sets of strings that Matplotlib uses as input parameters.

Matplotlib often uses simple data types like strings or tuples to define a
concept; e.g. the line capstyle can be specified as one of 'butt', 'round',
or 'projecting'. The classes in this module are used internally and serve to
document these concepts formally.

As an end-user you will not use these classes directly, but only the values
they define.
"""

from enum import Enum, auto
from numbers import Number

from matplotlib import _api, docstring


class _AutoStringNameEnum(Enum):
    """Automate the ``name = 'name'`` part of making a (str, Enum)."""

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __hash__(self):
        return str(self).__hash__()


def _deprecate_case_insensitive_join_cap(s):
    s_low = s.lower()
    if s != s_low:
        if s_low in ['miter', 'round', 'bevel']:
            _api.warn_deprecated(
                "3.3", message="Case-insensitive capstyles are deprecated "
                "since %(since)s and support for them will be removed "
                "%(removal)s; please pass them in lowercase.")
        elif s_low in ['butt', 'round', 'projecting']:
            _api.warn_deprecated(
                "3.3", message="Case-insensitive joinstyles are deprecated "
                "since %(since)s and support for them will be removed "
                "%(removal)s; please pass them in lowercase.")
        # Else, error out at the check_in_list stage.
    return s_low


class JoinStyle(str, _AutoStringNameEnum):
    """
    Define how the connection between two line segments is drawn.

    For a visual impression of each *JoinStyle*, `view these docs online
    <JoinStyle>`, or run `JoinStyle.demo`.

    Lines in Matplotlib are typically defined by a 1D `~.path.Path` and a
    finite ``linewidth``, where the underlying 1D `~.path.Path` represents the
    center of the stroked line.

    By default, `~.backend_bases.GraphicsContextBase` defines the boundaries of
    a stroked line to simply be every point within some radius,
    ``linewidth/2``, away from any point of the center line. However, this
    results in corners appearing "rounded", which may not be the desired
    behavior if you are drawing, for example, a polygon or pointed star.

    **Supported values:**

    .. rst-class:: value-list

        'miter'
            the "arrow-tip" style. Each boundary of the filled-in area will
            extend in a straight line parallel to the tangent vector of the
            centerline at the point it meets the corner, until they meet in a
            sharp point.
        'round'
            stokes every point within a radius of ``linewidth/2`` of the center
            lines.
        'bevel'
            the "squared-off" style. It can be thought of as a rounded corner
            where the "circular" part of the corner has been cut off.

    .. note::

        Very long miter tips are cut off (to form a *bevel*) after a
        backend-dependent limit called the "miter limit", which specifies the
        maximum allowed ratio of miter length to line width. For example, the
        PDF backend uses the default value of 10 specified by the PDF standard,
        while the SVG backend does not even specify the miter limit, resulting
        in a default value of 4 per the SVG specification. Matplotlib does not
        currently allow the user to adjust this parameter.

        A more detailed description of the effect of a miter limit can be found
        in the `Mozilla Developer Docs
        <https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-miterlimit>`_

    .. plot::
        :alt: Demo of possible JoinStyle's

        from matplotlib._enums import JoinStyle
        JoinStyle.demo()

    """

    miter = auto()
    round = auto()
    bevel = auto()

    def __init__(self, s):
        s = _deprecate_case_insensitive_join_cap(s)
        Enum.__init__(self)

    @staticmethod
    def demo():
        """Demonstrate how each JoinStyle looks for various join angles."""
        import numpy as np
        import matplotlib.pyplot as plt

        def plot_angle(ax, x, y, angle, style):
            phi = np.radians(angle)
            xx = [x + .5, x, x + .5*np.cos(phi)]
            yy = [y, y, y + .5*np.sin(phi)]
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
            ax.plot(xx, yy, lw=1, color='black')
            ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)

        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.set_title('Join style')
        for x, style in enumerate(['miter', 'round', 'bevel']):
            ax.text(x, 5, style)
            for y, angle in enumerate([20, 45, 60, 90, 120]):
                plot_angle(ax, x, y, angle, style)
                if x == 0:
                    ax.text(-1.3, y, f'{angle} degrees')
        ax.set_xlim(-1.5, 2.75)
        ax.set_ylim(-.5, 5.5)
        ax.set_axis_off()
        fig.show()


JoinStyle.input_description = "{" \
        + ", ".join([f"'{js.name}'" for js in JoinStyle]) \
        + "}"


class CapStyle(str, _AutoStringNameEnum):
    r"""
    Define how the two endpoints (caps) of an unclosed line are drawn.

    How to draw the start and end points of lines that represent a closed curve
    (i.e. that end in a `~.path.Path.CLOSEPOLY`) is controlled by the line's
    `JoinStyle`. For all other lines, how the start and end points are drawn is
    controlled by the *CapStyle*.

    For a visual impression of each *CapStyle*, `view these docs online
    <CapStyle>` or run `CapStyle.demo`.

    **Supported values:**

    .. rst-class:: value-list

        'butt'
            the line is squared off at its endpoint.
        'projecting'
            the line is squared off as in *butt*, but the filled in area
            extends beyond the endpoint a distance of ``linewidth/2``.
        'round'
            like *butt*, but a semicircular cap is added to the end of the
            line, of radius ``linewidth/2``.

    .. plot::
        :alt: Demo of possible CapStyle's

        from matplotlib._enums import CapStyle
        CapStyle.demo()

    """
    butt = 'butt'
    projecting = 'projecting'
    round = 'round'

    def __init__(self, s):
        s = _deprecate_case_insensitive_join_cap(s)
        Enum.__init__(self)

    @staticmethod
    def demo():
        """Demonstrate how each CapStyle looks for a thick line segment."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4, 1.2))
        ax = fig.add_axes([0, 0, 1, 0.8])
        ax.set_title('Cap style')

        for x, style in enumerate(['butt', 'round', 'projecting']):
            ax.text(x+0.25, 0.85, style, ha='center')
            xx = [x, x+0.5]
            yy = [0, 0]
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_capstyle=style)
            ax.plot(xx, yy, lw=1, color='black')
            ax.plot(xx, yy, 'o', color='tab:red', markersize=3)
        ax.text(2.25, 0.55, '(default)', ha='center')

        ax.set_ylim(-.5, 1.5)
        ax.set_axis_off()
        fig.show()


CapStyle.input_description = "{" \
        + ", ".join([f"'{cs.name}'" for cs in CapStyle]) \
        + "}"

docstring.interpd.update({'JoinStyle': JoinStyle.input_description,
                          'CapStyle': CapStyle.input_description})


#: Maps short codes for line style to their full name used by backends.
_ls_mapper = {'': 'None', ' ': 'None', 'none': 'None',
              '-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
_deprecated_lineStyles = {
    '-':    '_draw_solid',
    '--':   '_draw_dashed',
    '-.':   '_draw_dash_dot',
    ':':    '_draw_dotted',
    'None': '_draw_nothing',
    ' ':    '_draw_nothing',
    '':     '_draw_nothing',
}


class NamedLineStyle(str, _AutoStringNameEnum):
    """
    Describe if the line is solid or dashed, and the dash pattern, if any.

    All lines in Matplotlib are considered either solid or "dashed". Some
    common dashing patterns are built-in, and are sufficient for a majority of
    uses:

        ===============================   =================
        Linestyle                         Description
        ===============================   =================
        ``'-'`` or ``'solid'``            solid line
        ``'--'`` or  ``'dashed'``         dashed line
        ``'-.'`` or  ``'dashdot'``        dash-dotted line
        ``':'`` or ``'dotted'``           dotted line
        ``'None'`` or ``' '`` or ``''``   draw nothing
        ===============================   =================

    However, for more fine-grained control, one can directly specify the
    dashing pattern by specifying::

        (offset, onoffseq)

    where ``onoffseq`` is an even length tuple specifying the lengths of each
    subsequent dash and space, and ``offset`` controls at which point in this
    pattern the start of the line will begin (to allow you to e.g. prevent
    corners from happening to land in between dashes).

    For example, (5, 2, 1, 2) describes a sequence of 5 point and 1 point
    dashes separated by 2 point spaces.

    Setting ``onoffseq`` to ``None`` results in a solid *LineStyle*.

    The default dashing patterns described in the table above are themselves
    all described in this notation, and can therefore be customized by editing
    the appropriate ``lines.*_pattern`` *rc* parameter, as described in
    :doc:`/tutorials/introductory/customizing`.

    .. plot::
        :alt: Demo of possible LineStyle's.

        from matplotlib._types import LineStyle
        LineStyle.demo()

    .. note::

        In addition to directly taking a ``linestyle`` argument,
        `~.lines.Line2D` exposes a ``~.lines.Line2D.set_dashes`` method that
        can be used to create a new *LineStyle* by providing just the
        ``onoffseq``, but does not let you customize the offset. This method is
        called when using the keyword *dashes* to the cycler , as shown in
        :doc:`property_cycle </tutorials/intermediate/color_cycle>`.
    """
    solid = auto()
    dashed = auto()
    dotted = auto()
    dashdot = auto()
    none = auto()
    custom = auto()

class LineStyle(str):

    def __init__(self, ls, scale=1):
        """
        Parameters
        ----------
        ls : str or dash tuple
            A description of the dashing pattern of the line. Allowed string
            inputs are {'-', 'solid', '--', 'dashed', '-.', 'dashdot', ':',
            'dotted', '', ' ', 'None', 'none'}. Alternatively, the dash tuple
            (``offset``, ``onoffseq``) can be specified directly in points.
        scale : float
            Uniformly scale the internal dash sequence length by a constant
            factor.
        """

        self._linestyle_spec = ls
        if isinstance(ls, str):
            if ls in [' ', '', 'None']:
                ls = 'none'
            if ls in _ls_mapper:
                ls = _ls_mapper[ls]
            Enum.__init__(self)
            offset, onoffseq = None, None
        else:
            try:
                offset, onoffseq = ls
            except ValueError:  # not enough/too many values to unpack
                raise ValueError('LineStyle should be a string or a 2-tuple, '
                                 'instead received: ' + str(ls))
        if offset is None:
            _api.warn_deprecated(
                "3.3", message="Passing the dash offset as None is deprecated "
                "since %(since)s and support for it will be removed "
                "%(removal)s; pass it as zero instead.")
            offset = 0

        if onoffseq is not None:
            # normalize offset to be positive and shorter than the dash cycle
            dsum = sum(onoffseq)
            if dsum:
                offset %= dsum
            if len(onoffseq) % 2 != 0:
                raise ValueError('LineStyle onoffseq must be of even length.')
            if not all(isinstance(elem, Number) for elem in onoffseq):
                raise ValueError('LineStyle onoffseq must be list of floats.')
        self._us_offset = offset
        self._us_onoffseq = onoffseq

    def __hash__(self):
        if self == LineStyle.custom:
            return (self._us_offset, tuple(self._us_onoffseq)).__hash__()
        return _AutoStringNameEnum.__hash__(self)


    def get_dashes(self, lw=1):
        """
        Get the (scaled) dash sequence for this `.LineStyle`.
        """
        # defer lookup until draw time
        if self._us_offset is None or self._us_onoffseq is None:
            self._us_offset, self._us_onoffseq = \
                    LineStyle._get_dash_pattern(self.name)
            # normalize offset to be positive and shorter than the dash cycle
            dsum = sum(self._us_onoffseq)
            self._us_offset %= dsum
        return self._scale_dashes(self._us_offset, self._us_onoffseq, lw)

    @staticmethod
    def _scale_dashes(offset, dashes, lw):
        from . import rcParams
        if not rcParams['lines.scale_dashes']:
            return offset, dashes
        scaled_offset = offset * lw
        scaled_dashes = ([x * lw if x is not None else None for x in dashes]
                          if dashes is not None else None)
        return scaled_offset, scaled_dashes

    @staticmethod
    def _get_dash_pattern(style):
        """Convert linestyle string to explicit dash pattern."""
        # import must be local for validator code to live here
        from . import rcParams
        # un-dashed styles
        if style in ['solid', 'None']:
            offset = 0
            dashes = None
        # dashed styles
        elif style in ['dashed', 'dashdot', 'dotted']:
            offset = 0
            dashes = tuple(rcParams['lines.{}_pattern'.format(style)])
        return offset, dashes

    @staticmethod
    def from_dashes(seq):
        """
        Create a `.LineStyle` from a dash sequence (i.e. the ``onoffseq``).

        The dash sequence is a sequence of floats of even length describing
        the length of dashes and spaces in points.

        Parameters
        ----------
        seq : sequence of floats (on/off ink in points) or (None, None)
            If *seq* is empty or ``(None, None)``, the `.LineStyle` will be
            solid.
        """
        if seq == (None, None) or len(seq) == 0:
            return LineStyle('-')
        else:
            return LineStyle((0, seq))

    @staticmethod
    def demo():
        import numpy as np
        import matplotlib.pyplot as plt

        linestyle_str = [
            ('solid', 'solid'),      # Same as (0, ()) or '-'
            ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
            ('dashed', 'dashed'),    # Same as '--'
            ('dashdot', 'dashdot')]  # Same as '-.'

        linestyle_tuple = [
            ('loosely dotted',        (0, (1, 10))),
            ('dotted',                (0, (1, 1))),
            ('densely dotted',        (0, (1, 1))),

            ('loosely dashed',        (0, (5, 10))),
            ('dashed',                (0, (5, 5))),
            ('densely dashed',        (0, (5, 1))),

            ('loosely dashdotted',    (0, (3, 10, 1, 10))),
            ('dashdotted',            (0, (3, 5, 1, 5))),
            ('densely dashdotted',    (0, (3, 1, 1, 1))),

            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

        def plot_linestyles(ax, linestyles, title):
            X, Y = np.linspace(0, 100, 10), np.zeros(10)
            yticklabels = []

            for i, (name, linestyle) in enumerate(linestyles):
                ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5,
                        color='black')
                yticklabels.append(name)

            ax.set_title(title)
            ax.set(ylim=(-0.5, len(linestyles)-0.5),
                   yticks=np.arange(len(linestyles)),
                   yticklabels=yticklabels)
            ax.tick_params(left=False, bottom=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # For each line style, add a text annotation with a small offset
            # from the reference point (0 in Axes coords, y tick value in Data
            # coords).
            for i, (name, linestyle) in enumerate(linestyles):
                ax.annotate(repr(linestyle),
                            xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                            xytext=(-6, -12), textcoords='offset points',
                            color="blue", fontsize=8, ha="right",
                            family="monospace")

        ax0, ax1 = (plt.figure(figsize=(10, 8))
                    .add_gridspec(2, 1, height_ratios=[1, 3])
                    .subplots())

        plot_linestyles(ax0, linestyle_str[::-1], title='Named linestyles')
        plot_linestyles(ax1, linestyle_tuple[::-1],
                        title='Parametrized linestyles')

        plt.tight_layout()
        plt.show()


LineStyle._ls_mapper = _ls_mapper
LineStyle._deprecated_lineStyles = _deprecated_lineStyles
