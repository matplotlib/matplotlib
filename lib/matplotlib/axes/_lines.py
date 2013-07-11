"""
Lines and spans
"""

from matplotlib import docstring
from matplotlib import transforms as mtransforms
from matplotlib import line as mlines


@docstring.dedent_interpd
def axhline(ax, y=0, xmin=0, xmax=1, **kwargs):
    """
    Add a horizontal line across the axis.

    Parameters
    ----------
    y : scalar, optional, default: 0
        y position in data coordinates of the horizontal line.

    xmin : scalar, optional, default: 0
        Should be between 0 and 1, 0 being the far left of the plot, 1 the
        far right of the plot.

    xmax : scalar, optional, default: 1
        Should be between 0 and 1, 0 being the far left of the plot, 1 the
        far right of the plot.

    Returns
    -------
    `~matplotlib.lines.Line2D`

    Notes
    -----
    kwargs are the same as kwargs to plot, and can be
    used to control the line properties.  e.g.,

    Examples
    --------

    * draw a thick red hline at 'y' = 0 that spans the xrange::

        >>> axhline(linewidth=4, color='r')

    * draw a default hline at 'y' = 1 that spans the xrange::

        >>> axhline(y=1)

    * draw a default hline at 'y' = .5 that spans the the middle half of
        the xrange::

        >>> axhline(y=.5, xmin=0.25, xmax=0.75)

    Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
    with the exception of 'transform':

    %(Line2D)s

    See also
    --------
    `axhspan` for example plot and source code
    """

    if "transform" in kwargs:
        raise ValueError(
            "'transform' is not allowed as a kwarg;"
            + "axhline generates its own transform.")
    ymin, ymax = ax.get_ybound()

    # We need to strip away the units for comparison with
    # non-unitized bounds
    ax._process_unit_info(ydata=y, kwargs=kwargs)
    yy = ax.convert_yunits(y)
    scaley = (yy < ymin) or (yy > ymax)

    trans = mtransforms.blended_transform_factory(
        ax.transAxes, ax.transData)
    l = mlines.Line2D([xmin, xmax], [y, y], transform=trans, **kwargs)
    ax.add_line(l)
    ax.autoscale_view(scalex=False, scaley=scaley)
    return l
