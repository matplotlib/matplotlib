"""
Lines and spans
"""

from matplotlib import docstring
from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines


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


def hlines(ax, y, xmin, xmax, colors='k', linestyles='solid',
            label='', **kwargs):
    """
    Plot horizontal lines at each `y` from `xmin` to `xmax`.

    Parameters
    ----------
    y : scalar or sequence of scalar
        y-indexes where to plot the lines.

    xmin, xmax : scalar or 1D array_like
        Respective beginning and end of each line. If scalars are
        provided, all lines will have same length.

    colors : array_like of colors, optional, default: 'k'

    linestyles : ['solid' | 'dashed' | 'dashdot' | 'dotted'], optional

    label : string, optional, default: ''

    Returns
    -------
    lines : `~matplotlib.collections.LineCollection`

    Other parameters
    ----------------
    kwargs :  `~matplotlib.collections.LineCollection` properties.

    See also
    --------
    vlines : vertical lines

    Examples
    --------
    .. plot:: mpl_examples/pylab_examples/vline_hline_demo.py

    """

    # We do the conversion first since not all unitized data is uniform
    # process the unit information
    ax._process_unit_info([xmin, xmax], y, kwargs=kwargs)
    y = ax.convert_yunits(y)
    xmin = ax.convert_xunits(xmin)
    xmax = ax.convert_xunits(xmax)

    if not iterable(y):
        y = [y]
    if not iterable(xmin):
        xmin = [xmin]
    if not iterable(xmax):
        xmax = [xmax]

    y = np.asarray(y)
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax)

    if len(xmin) == 1:
        xmin = np.resize(xmin, y.shape)
    if len(xmax) == 1:
        xmax = np.resize(xmax, y.shape)

    if len(xmin) != len(y):
        raise ValueError('xmin and y are unequal sized sequences')
    if len(xmax) != len(y):
        raise ValueError('xmax and y are unequal sized sequences')

    verts = [((thisxmin, thisy), (thisxmax, thisy))
                for thisxmin, thisxmax, thisy in zip(xmin, xmax, y)]
    coll = mcoll.LineCollection(verts, colors=colors,
                                linestyles=linestyles, label=label)
    ax.add_collection(coll)
    coll.update(kwargs)

    if len(y) > 0:
        minx = min(xmin.min(), xmax.min())
        maxx = max(xmin.max(), xmax.max())
        miny = y.min()
        maxy = y.max()

        corners = (minx, miny), (maxx, maxy)

        ax.update_datalim(corners)
        ax.autoscale_view()

    return coll
