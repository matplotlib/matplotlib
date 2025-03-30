"""Helpers to create new Figures."""

from matplotlib import is_interactive

from ._figure import Figure
from ._promotion import promote_figure


def figure(
    *,
    label=None,  # autoincrement if None, else integer from 1-N
    figsize=None,  # defaults to rc figure.figsize
    dpi=None,  # defaults to rc figure.dpi
    facecolor=None,  # defaults to rc figure.facecolor
    edgecolor=None,  # defaults to rc figure.edgecolor
    frameon=True,
    FigureClass=Figure,
    clear=False,
    auto_draw=True,
    **kwargs,
):
    """
    Create a new figure

    Parameters
    ----------
    label : str, optional
        Label for the figure.  Will be used as the window title

    figsize : (float, float), default: :rc:`figure.figsize`
        Width, height in inches.

    dpi : float, default: :rc:`figure.dpi`
        The resolution of the figure in dots-per-inch.

    facecolor : color, default: :rc:`figure.facecolor`
        The background color.

    edgecolor : color, default: :rc:`figure.edgecolor`
        The border color.

    frameon : bool, default: True
        If False, suppress drawing the figure frame.

    FigureClass : subclass of `~matplotlib.figure.Figure`
        Optionally use a custom `~matplotlib.figure.Figure` instance.

    tight_layout : bool or dict, default: :rc:`figure.autolayout`
        If ``False`` use *subplotpars*. If ``True`` adjust subplot parameters
        using `~matplotlib.figure.Figure.tight_layout` with default padding.
        When providing a dict containing the keys ``pad``, ``w_pad``,
        ``h_pad``, and ``rect``, the default
        `~matplotlib.figure.Figure.tight_layout` paddings will be overridden.

    **kwargs : optional
        See `~.matplotlib.figure.Figure` for other possible arguments.

    Returns
    -------
    `~matplotlib.figure.Figure`
        The `~matplotlib.figure.Figure` instance returned will also be passed
        to new_figure_manager in the backends, which allows to hook custom
        `~matplotlib.figure.Figure` classes into the pyplot
        interface. Additional kwargs will be passed to the
        `~matplotlib.figure.Figure` init function.

    """

    fig = FigureClass(
        label=label,
        figsize=figsize,
        dpi=dpi,
        facecolor=facecolor,
        edgecolor=edgecolor,
        frameon=frameon,
        **kwargs,
    )
    if is_interactive():
        promote_figure(fig, auto_draw=auto_draw)
    return fig


def subplots(
    nrows=1,
    ncols=1,
    *,
    sharex=False,
    sharey=False,
    squeeze=True,
    subplot_kw=None,
    gridspec_kw=None,
    **fig_kw,
):
    """
    Create a figure and a set of subplots.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    nrows, ncols : int, default: 1
        Number of rows/columns of the subplot grid.

    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (*sharex*) or y (*sharey*)
        axes:

        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.

        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are created. Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are created. To later turn other subplots' ticklabels
        on, use `~matplotlib.axes.Axes.tick_params`.

        When subplots have a shared axis that has units, calling
        `~matplotlib.axis.Axis.set_units` will update each axis with the
        new units.

    squeeze : bool, default: True
        - If True, extra dimensions are squeezed out from the returned
          array of `~matplotlib.axes.Axes`:

          - if only one subplot is constructed (nrows=ncols=1), the
            resulting single Axes object is returned as a scalar.
          - for Nx1 or 1xM subplots, the returned object is a 1D numpy
            object array of Axes objects.
          - for NxM, subplots with N>1 and M>1 are returned as a 2D array.

        - If False, no squeezing at all is done: the returned Axes object is
          always a 2D array containing Axes instances, even if it ends up
          being 1x1.

    subplot_kw : dict, optional
        Dict with keywords passed to the
        `~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.

    gridspec_kw : dict, optional
        Dict with keywords passed to the `~matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.

    **fig_kw
        All additional keyword arguments are passed to the
        `.figure` call.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`

    ax : `~matplotlib.axes.Axes` or array of Axes
        *ax* can be either a single `~matplotlib.axes.Axes` object or an
        array of Axes objects if more than one subplot was created.  The
        dimensions of the resulting array can be controlled with the squeeze
        keyword, see above.

        Typical idioms for handling the return value are::

            # using the variable ax for single a Axes
            fig, ax = plt.subplots()

            # using the variable axs for multiple Axes
            fig, axs = plt.subplots(2, 2)

            # using tuple unpacking for multiple Axes
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        The names ``ax`` and pluralized ``axs`` are preferred over ``axes``
        because for the latter it's not clear if it refers to a single
        `~matplotlib.axes.Axes` instance or a collection of these.

    See Also
    --------

    matplotlib.figure.Figure.subplots
    matplotlib.figure.Figure.add_subplot

    Examples
    --------
    ::

        # First create some toy data:
        x = np.linspace(0, 2*np.pi, 400)
        y = np.sin(x**2)

        # Create just a figure and only one subplot
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title('Simple plot')

        # Create two subplots and unpack the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(x, y)
        ax1.set_title('Sharing Y axis')
        ax2.scatter(x, y)

        # Create four polar axes and access them through the returned array
        fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
        axs[0, 0].plot(x, y)
        axs[1, 1].scatter(x, y)

        # Share a X axis with each column of subplots
        plt.subplots(2, 2, sharex='col')

        # Share a Y axis with each row of subplots
        plt.subplots(2, 2, sharey='row')

        # Share both X and Y axes with all subplots
        plt.subplots(2, 2, sharex='all', sharey='all')

        # Note that this is the same as
        plt.subplots(2, 2, sharex=True, sharey=True)

        # Create figure number 10 with a single subplot
        # and clears it if it already exists.
        fig, ax = plt.subplots(num=10, clear=True)

    """
    fig = figure(**fig_kw)
    axs = fig.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
    )
    return fig, axs


def subplot_mosaic(
    layout, *, subplot_kw=None, gridspec_kw=None, empty_sentinel=".", **fig_kw
):
    """
    Build a layout of Axes based on ASCII art or nested lists.

    This is a helper function to build complex `~matplotlib.gridspec.GridSpec`
    layouts visually.

    .. note ::

       This API is provisional and may be revised in the future based on
       early user feedback.


    Parameters
    ----------
    layout : list of list of {hashable or nested} or str

        A visual layout of how you want your Axes to be arranged
        labeled as strings.  For example ::

           x = [['A panel', 'A panel', 'edge'],
                ['C panel', '.',       'edge']]

        Produces 4 axes:

        - 'A panel' which is 1 row high and spans the first two columns
        - 'edge' which is 2 rows high and is on the right edge
        - 'C panel' which in 1 row and 1 column wide in the bottom left
        - a blank space 1 row and 1 column wide in the bottom center

        Any of the entries in the layout can be a list of lists
        of the same form to create nested layouts.

        If input is a str, then it must be of the form ::

          '''
          AAE
          C.E
          '''

        where each character is a column and each line is a row.
        This only allows only single character Axes labels and does
        not allow nesting but is very terse.

    subplot_kw : dict, optional
        Dictionary with keywords passed to the `~matplotlib.figure.Figure.add_subplot`
        call used to create each subplot.

    gridspec_kw : dict, optional
        Dictionary with keywords passed to the `~matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.

    empty_sentinel : object, optional
        Entry in the layout to mean "leave this space empty".  Defaults
        to ``'.'``. Note, if *layout* is a string, it is processed via
        `inspect.cleandoc` to remove leading white space, which may
        interfere with using white-space as the empty sentinel.

    **fig_kw
        All additional keyword arguments are passed to the
        `.figure` call.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
       The new figure

    dict[label, Axes]
       A dictionary mapping the labels to the Axes objects.  The order of
       the axes is left-to-right and top-to-bottom of their position in the
       total layout.

    """
    fig = figure(**fig_kw)
    ax_dict = fig.subplot_mosaic(
        layout,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
        empty_sentinel=empty_sentinel,
    )
    return fig, ax_dict
