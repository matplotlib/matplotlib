"""
Provides a MATLAB-like plotting framework.

:mod:`~matplotlib.pylab` combines pyplot with numpy into a single namespace.
This is convenient for interactive work, but for programming it
is recommended that the namespaces be kept separate, e.g.::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1);
    y = np.sin(x)
    plt.plot(x, y)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import inspect
import re
import sys
import warnings

from cycler import cycler
import matplotlib
from matplotlib import (
    # submodules
    _pylab_helpers, cm, docstring, mlab, style,
    # functions
    get_backend, interactive, rc, rc_context,
    # objects
    rcParams, rcParamsDefault)
from matplotlib.artist import getp, get, setp, Artist
from matplotlib.axes import Axes, Subplot
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.cbook import (
    _string_to_bool, dedent, deprecated, is_numlike, silent_list)
from matplotlib.cm import get_cmap, register_cmap
from matplotlib.figure import Figure, figaspect
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread, imsave
from matplotlib.projections import PolarAxes
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.scale import get_scale_docs, get_scale_names

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import SubplotTool, Button, Slider, Widget

from .ticker import TickHelper, Formatter, FixedFormatter, NullFormatter,\
           FuncFormatter, FormatStrFormatter, ScalarFormatter,\
           LogFormatter, LogFormatterExponent, LogFormatterMathtext,\
           Locator, IndexLocator, FixedLocator, NullLocator,\
           LinearLocator, LogLocator, AutoLocator, MultipleLocator,\
           MaxNLocator
from matplotlib.backends import pylab_setup

## Backend detection ##
def _backend_selection():
    """ If rcParams['backend_fallback'] is true, check to see if the
        current backend is compatible with the current running event
        loop, and if not switches to a compatible one.
    """
    backend = rcParams['backend']
    if not rcParams['backend_fallback'] or \
                     backend not in _interactive_bk:
        return
    is_agg_backend = rcParams['backend'].endswith('Agg')
    if 'wx' in sys.modules and not backend in ('WX', 'WXAgg'):
        import wx
        if wx.App.IsMainLoopRunning():
            rcParams['backend'] = 'wx' + 'Agg' * is_agg_backend
    elif 'PyQt4.QtCore' in sys.modules and not backend == 'Qt4Agg':
        import PyQt4.QtGui
        if not PyQt4.QtGui.qApp.startingUp():
            # The mainloop is running.
            rcParams['backend'] = 'qt4Agg'
    elif 'PyQt5.QtCore' in sys.modules and not backend == 'Qt5Agg':
        import PyQt5.QtWidgets
        if not PyQt5.QtWidgets.qApp.startingUp():
            # The mainloop is running.
            rcParams['backend'] = 'qt5Agg'
    elif ('gtk' in sys.modules and
          backend not in ('GTK', 'GTKAgg', 'GTKCairo')):
        if 'gi' in sys.modules:
            from gi.repository import GObject
            ml = GObject.MainLoop
        else:
            import gobject
            ml = gobject.MainLoop
        if ml().is_running():
            rcParams['backend'] = 'gtk' + 'Agg' * is_agg_backend
    elif 'Tkinter' in sys.modules and not backend == 'TkAgg':
        # import Tkinter
        pass  # what if anything do we need to do for tkinter?

_backend_selection()

## Global ##

_backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()

_IP_REGISTERED = None
_INSTALL_FIG_OBSERVER = False


def install_repl_displayhook():
    """
    Install a repl display hook so that any stale figure are automatically
    redrawn when control is returned to the repl.

    This works with IPython terminals and kernels,
    as well as vanilla python shells.
    """
    global _IP_REGISTERED
    global _INSTALL_FIG_OBSERVER

    class _NotIPython(Exception):
        pass

    # see if we have IPython hooks around, if use them

    try:
        if 'IPython' in sys.modules:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                raise _NotIPython()

            if _IP_REGISTERED:
                return

            def post_execute():
                if matplotlib.is_interactive():
                    draw_all()

            # IPython >= 2
            try:
                ip.events.register('post_execute', post_execute)
            except AttributeError:
                # IPython 1.x
                ip.register_post_execute(post_execute)

            _IP_REGISTERED = post_execute
            _INSTALL_FIG_OBSERVER = False

            # trigger IPython's eventloop integration, if available
            from IPython.core.pylabtools import backend2gui

            ipython_gui_name = backend2gui.get(get_backend())
            if ipython_gui_name:
                ip.enable_gui(ipython_gui_name)
        else:
            _INSTALL_FIG_OBSERVER = True

    # import failed or ipython is not running
    except (ImportError, _NotIPython):
        _INSTALL_FIG_OBSERVER = True


def uninstall_repl_displayhook():
    """
    Uninstalls the matplotlib display hook.

    .. warning

       Need IPython >= 2 for this to work.  For IPython < 2 will raise a
       ``NotImplementedError``

    .. warning

       If you are using vanilla python and have installed another
       display hook this will reset ``sys.displayhook`` to what ever
       function was there when matplotlib installed it's displayhook,
       possibly discarding your changes.
    """
    global _IP_REGISTERED
    global _INSTALL_FIG_OBSERVER
    if _IP_REGISTERED:
        from IPython import get_ipython
        ip = get_ipython()
        try:
            ip.events.unregister('post_execute', _IP_REGISTERED)
        except AttributeError:
            raise NotImplementedError("Can not unregister events "
                                      "in IPython < 2.0")
        _IP_REGISTERED = None

    if _INSTALL_FIG_OBSERVER:
        _INSTALL_FIG_OBSERVER = False


draw_all = _pylab_helpers.Gcf.draw_all


@docstring.copy_dedent(Artist.findobj)
def findobj(o=None, match=None, include_self=True):
    if o is None:
        o = gcf()
    return o.findobj(match, include_self=include_self)


def switch_backend(newbackend):
    """
    Switch the default backend.  This feature is **experimental**, and
    is only expected to work switching to an image backend.  e.g., if
    you have a bunch of PostScript scripts that you want to run from
    an interactive ipython session, you may want to switch to the PS
    backend before running them to avoid having a bunch of GUI windows
    popup.  If you try to interactively switch from one GUI backend to
    another, you will explode.

    Calling this command will close all open windows.
    """
    close('all')
    global _backend_mod, new_figure_manager, draw_if_interactive, _show
    matplotlib.use(newbackend, warn=False, force=True)
    from matplotlib.backends import pylab_setup
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()


def show(*args, **kw):
    """
    Display a figure.
    When running in ipython with its pylab mode, display all
    figures and return to the ipython prompt.

    In non-interactive mode, display all figures and block until
    the figures have been closed; in interactive mode it has no
    effect unless figures were created prior to a change from
    non-interactive to interactive mode (not recommended).  In
    that case it displays the figures but does not block.

    A single experimental keyword argument, *block*, may be
    set to True or False to override the blocking behavior
    described above.
    """
    global _show
    return _show(*args, **kw)


def isinteractive():
    """Return status of interactive mode."""
    return matplotlib.is_interactive()


def ioff():
    """Turn interactive mode off."""
    matplotlib.interactive(False)
    uninstall_repl_displayhook()


def ion():
    """Turn interactive mode on."""
    matplotlib.interactive(True)
    install_repl_displayhook()


def pause(interval):
    """
    Pause for *interval* seconds.

    If there is an active figure it will be updated and displayed,
    and the GUI event loop will run during the pause.

    If there is no active figure, or if a non-interactive backend
    is in use, this executes time.sleep(interval).

    This can be used for crude animation. For more complex
    animation, see :mod:`matplotlib.animation`.

    This function is experimental; its behavior may be changed
    or extended in a future release.

    """
    backend = rcParams['backend']
    if backend in _interactive_bk:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            show(block=False)
            canvas.start_event_loop(interval)
            return

    # No on-screen figure is active, so sleep() is all we need.
    import time
    time.sleep(interval)


@docstring.copy_dedent(matplotlib.rcdefaults)
def rcdefaults():
    matplotlib.rcdefaults()
    if matplotlib.is_interactive():
        draw_all()


# This function is not autogenerated because it is needed in when
# autogenerating other functions, see `_post_processors`.
def sci(im):
    """
    Set the current image.  This image will be the target of colormap
    commands like :func:`~matplotlib.pyplot.jet`,
    :func:`~matplotlib.pyplot.hot` or
    :func:`~matplotlib.pyplot.clim`).  The current image is an
    attribute of the current axes.
    """
    gca()._sci(im)


## Any Artist ##


def xkcd(scale=1, length=100, randomness=2):
    """
    Turns on `xkcd <https://xkcd.com/>`_ sketch-style drawing mode.
    This will only have effect on things drawn after this function is
    called.

    For best results, the "Humor Sans" font should be installed: it is
    not included with matplotlib.

    Parameters
    ----------
    scale : float, optional
        The amplitude of the wiggle perpendicular to the source line.
    length : float, optional
        The length of the wiggle along the line.
    randomness : float, optional
        The scale factor by which the length is shrunken or expanded.

    Notes
    -----
    This function works by a number of rcParams, so it will probably
    override others you have set before.

    If you want the effects of this function to be temporary, it can
    be used as a context manager, for example::

        with plt.xkcd():
            # This figure will be in XKCD-style
            fig1 = plt.figure()
            # ...

        # This figure will be in regular style
        fig2 = plt.figure()
    """
    if rcParams['text.usetex']:
        raise RuntimeError(
            "xkcd mode is not compatible with text.usetex = True")

    from matplotlib import patheffects
    context = rc_context()
    try:
        rcParams['font.family'] = ['xkcd', 'Humor Sans', 'Comic Sans MS']
        rcParams['font.size'] = 14.0
        rcParams['path.sketch'] = (scale, length, randomness)
        rcParams['path.effects'] = [
            patheffects.withStroke(linewidth=4, foreground="w")]
        rcParams['axes.linewidth'] = 1.5
        rcParams['lines.linewidth'] = 2.0
        rcParams['figure.facecolor'] = 'white'
        rcParams['grid.linewidth'] = 0.0
        rcParams['axes.grid'] = False
        rcParams['axes.unicode_minus'] = False
        rcParams['axes.edgecolor'] = 'black'
        rcParams['xtick.major.size'] = 8
        rcParams['xtick.major.width'] = 3
        rcParams['ytick.major.size'] = 8
        rcParams['ytick.major.width'] = 3
    except:
        context.__exit__(*sys.exc_info())
        raise
    return context


## Figures ##

def figure(num=None,  # autoincrement if None, else integer from 1-N
           figsize=None,  # defaults to rc figure.figsize
           dpi=None,  # defaults to rc figure.dpi
           facecolor=None,  # defaults to rc figure.facecolor
           edgecolor=None,  # defaults to rc figure.edgecolor
           frameon=True,
           FigureClass=Figure,
           clear=False,
           **kwargs
           ):
    """
    Creates a new figure.

    Parameters
    ----------

    num : integer or string, optional, default: none
        If not provided, a new figure will be created, and the figure number
        will be incremented. The figure objects holds this number in a `number`
        attribute.
        If num is provided, and a figure with this id already exists, make
        it active, and returns a reference to it. If this figure does not
        exists, create it and returns it.
        If num is a string, the window title will be set to this figure's
        `num`.

    figsize : tuple of integers, optional, default: None
        width, height in inches. If not provided, defaults to rc
        figure.figsize.

    dpi : integer, optional, default: None
        resolution of the figure. If not provided, defaults to rc figure.dpi.

    facecolor :
        the background color. If not provided, defaults to rc figure.facecolor.

    edgecolor :
        the border color. If not provided, defaults to rc figure.edgecolor.

    frameon : bool, optional, default: True
        If False, suppress drawing the figure frame.

    FigureClass : class derived from matplotlib.figure.Figure
        Optionally use a custom Figure instance.

    clear : bool, optional, default: False
        If True and the figure already exists, then it is cleared.

    Returns
    -------
    figure : Figure
        The Figure instance returned will also be passed to new_figure_manager
        in the backends, which allows to hook custom Figure classes into the
        pylab interface. Additional kwargs will be passed to the figure init
        function.

    Notes
    -----
    If you are creating many figures, make sure you explicitly call "close"
    on the figures you are not using, because this will enable pylab
    to properly clean up the memory.

    rcParams defines the default values, which can be modified in the
    matplotlibrc file

    """

    if figsize is None:
        figsize = rcParams['figure.figsize']
    if dpi is None:
        dpi = rcParams['figure.dpi']
    if facecolor is None:
        facecolor = rcParams['figure.facecolor']
    if edgecolor is None:
        edgecolor = rcParams['figure.edgecolor']

    allnums = get_fignums()
    next_num = max(allnums) + 1 if allnums else 1
    figLabel = ''
    if num is None:
        num = next_num
    elif isinstance(num, six.string_types):
        figLabel = num
        allLabels = get_figlabels()
        if figLabel not in allLabels:
            if figLabel == 'all':
                warnings.warn("close('all') closes all existing figures")
            num = next_num
        else:
            inum = allLabels.index(figLabel)
            num = allnums[inum]
    else:
        num = int(num)  # crude validation of num argument

    figManager = _pylab_helpers.Gcf.get_fig_manager(num)
    if figManager is None:
        max_open_warning = rcParams['figure.max_open_warning']

        if (max_open_warning >= 1 and len(allnums) >= max_open_warning):
            warnings.warn(
                "More than %d figures have been opened. Figures "
                "created through the pyplot interface "
                "(`matplotlib.pyplot.figure`) are retained until "
                "explicitly closed and may consume too much memory. "
                "(To control this warning, see the rcParam "
                "`figure.max_open_warning`)." %
                max_open_warning, RuntimeWarning)

        if get_backend().lower() == 'ps':
            dpi = 72

        figManager = new_figure_manager(num, figsize=figsize,
                                        dpi=dpi,
                                        facecolor=facecolor,
                                        edgecolor=edgecolor,
                                        frameon=frameon,
                                        FigureClass=FigureClass,
                                        **kwargs)

        if figLabel:
            figManager.set_window_title(figLabel)
            figManager.canvas.figure.set_label(figLabel)

        # make this figure current on button press event
        def make_active(event):
            _pylab_helpers.Gcf.set_active(figManager)

        cid = figManager.canvas.mpl_connect('button_press_event', make_active)
        figManager._cidgcf = cid

        _pylab_helpers.Gcf.set_active(figManager)
        fig = figManager.canvas.figure
        fig.number = num

        # make sure backends (inline) that we don't ship that expect this
        # to be called in plotting commands to make the figure call show
        # still work.  There is probably a better way to do this in the
        # FigureManager base class.
        if matplotlib.is_interactive():
            draw_if_interactive()

        if _INSTALL_FIG_OBSERVER:
            fig.stale_callback = _auto_draw_if_interactive

    if clear:
        figManager.canvas.figure.clear()

    return figManager.canvas.figure


def _auto_draw_if_interactive(fig, val):
    """
    This is an internal helper function for making sure that auto-redrawing
    works as intended in the plain python repl.

    Parameters
    ----------
    fig : Figure
        A figure object which is assumed to be associated with a canvas
    """
    if val and matplotlib.is_interactive() and not fig.canvas.is_saving():
        fig.canvas.draw_idle()


def gcf():
    "Get a reference to the current figure."

    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is not None:
        return figManager.canvas.figure
    else:
        return figure()


def fignum_exists(num):
    return _pylab_helpers.Gcf.has_fignum(num) or num in get_figlabels()


def get_fignums():
    """Return a list of existing figure numbers."""
    return sorted(_pylab_helpers.Gcf.figs)


def get_figlabels():
    "Return a list of existing figure labels."
    figManagers = _pylab_helpers.Gcf.get_all_fig_managers()
    figManagers.sort(key=lambda m: m.num)
    return [m.canvas.figure.get_label() for m in figManagers]


def get_current_fig_manager():
    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is None:
        gcf()  # creates an active figure as a side effect
        figManager = _pylab_helpers.Gcf.get_active()
    return figManager


def close(*args):
    """
    Close a figure window.

    ``close()`` by itself closes the current figure

    ``close(h)`` where *h* is a :class:`Figure` instance, closes that figure

    ``close(num)`` closes figure number *num*

    ``close(name)`` where *name* is a string, closes figure with that label

    ``close('all')`` closes all the figure windows
    """

    if len(args) == 0:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is None:
            return
        else:
            _pylab_helpers.Gcf.destroy(figManager.num)
    elif len(args) == 1:
        arg = args[0]
        if arg == 'all':
            _pylab_helpers.Gcf.destroy_all()
        elif isinstance(arg, six.integer_types):
            _pylab_helpers.Gcf.destroy(arg)
        elif hasattr(arg, 'int'):
            # if we are dealing with a type UUID, we
            # can use its integer representation
            _pylab_helpers.Gcf.destroy(arg.int)
        elif isinstance(arg, six.string_types):
            allLabels = get_figlabels()
            if arg in allLabels:
                num = get_fignums()[allLabels.index(arg)]
                _pylab_helpers.Gcf.destroy(num)
        elif isinstance(arg, Figure):
            _pylab_helpers.Gcf.destroy_fig(arg)
        else:
            raise TypeError(
                'Unrecognized argument type %s to close' % type(arg))
    else:
        raise TypeError('close takes 0 or 1 arguments')


## Axes ##


def axes(*args, **kwargs):
    """
    Add an axes to the figure.

    The axes is added at position *rect* specified by:

    - ``axes()`` by itself creates a default full ``subplot(111)`` window axis.

    - ``axes(rect, facecolor='w')`` where *rect* = [left, bottom, width,
      height] in normalized (0, 1) units.  *facecolor* is the background
      color for the axis, default white.

    - ``axes(h)`` where *h* is an axes instance makes *h* the current
      axis and the parent of *h* the current figure.
      An :class:`~matplotlib.axes.Axes` instance is returned.

    =========   ==============   ==============================================
    kwarg       Accepts          Description
    =========   ==============   ==============================================
    facecolor   color            the axes background color
    frameon     [True|False]     display the frame?
    sharex      otherax          current axes shares xaxis attribute
                                 with otherax
    sharey      otherax          current axes shares yaxis attribute
                                 with otherax
    polar       [True|False]     use a polar axes?
    aspect      [str | num]      ['equal', 'auto'] or a number.  If a number
                                 the ratio of x-unit/y-unit in screen-space.
                                 Also see
                                 :meth:`~matplotlib.axes.Axes.set_aspect`.
    =========   ==============   ==============================================

    Examples:

    * :file:`examples/pylab_examples/axes_demo.py` places custom axes.
    * :file:`examples/pylab_examples/shared_axis_demo.py` uses
      *sharex* and *sharey*.

    """

    nargs = len(args)
    if len(args) == 0:
        return subplot(111, **kwargs)
    if nargs > 1:
        raise TypeError('Only one non keyword arg to axes allowed')
    arg = args[0]

    if isinstance(arg, Axes):
        sca(arg)
        a = arg
    else:
        rect = arg
        a = gcf().add_axes(rect, **kwargs)
    return a


def delaxes(*args):
    """
    Remove an axes from the current figure.  If *ax*
    doesn't exist, an error will be raised.

    ``delaxes()``: delete the current axes
    """
    if not len(args):
        ax = gca()
    else:
        ax = args[0]
    ret = gcf().delaxes(ax)
    return ret


def sca(ax):
    """
    Set the current Axes instance to *ax*.

    The current Figure is updated to the parent of *ax*.
    """
    managers = _pylab_helpers.Gcf.get_all_fig_managers()
    for m in managers:
        if ax in m.canvas.figure.axes:
            _pylab_helpers.Gcf.set_active(m)
            m.canvas.figure.sca(ax)
            return
    raise ValueError("Axes instance argument was not found in a figure.")


# More ways of creating axes:


def subplot(*args, **kwargs):
    """
    Return a subplot axes positioned by the given grid definition.

    Typical call signature::

      subplot(nrows, ncols, plot_number)

    Where *nrows* and *ncols* are used to notionally split the figure
    into ``nrows * ncols`` sub-axes, and *plot_number* is used to identify
    the particular subplot that this function is to create within the notional
    grid. *plot_number* starts at 1, increments across rows first and has a
    maximum of ``nrows * ncols``.

    In the case when *nrows*, *ncols* and *plot_number* are all less than 10,
    a convenience exists, such that the a 3 digit number can be given instead,
    where the hundreds represent *nrows*, the tens represent *ncols* and the
    units represent *plot_number*. For instance::

      subplot(211)

    produces a subaxes in a figure which represents the top plot (i.e. the
    first) in a 2 row by 1 column notional grid (no grid actually exists,
    but conceptually this is how the returned subplot has been positioned).

    .. note::

       Creating a subplot will delete any pre-existing subplot that overlaps
       with it beyond sharing a boundary::

          import matplotlib.pyplot as plt
          # plot a line, implicitly creating a subplot(111)
          plt.plot([1,2,3])
          # now create a subplot which represents the top plot of a grid
          # with 2 rows and 1 column. Since this subplot will overlap the
          # first, the plot (and its axes) previously created, will be removed
          plt.subplot(211)
          plt.plot(range(12))
          plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background

       If you do not want this behavior, use the
       :meth:`~matplotlib.figure.Figure.add_subplot` method or the
       :func:`~matplotlib.pyplot.axes` function instead.

    Keyword arguments:

      *facecolor*:
        The background color of the subplot, which can be any valid
        color specifier.  See :mod:`matplotlib.colors` for more
        information.

      *polar*:
        A boolean flag indicating whether the subplot plot should be
        a polar projection.  Defaults to *False*.

      *projection*:
        A string giving the name of a custom projection to be used
        for the subplot. This projection must have been previously
        registered. See :mod:`matplotlib.projections`.

    .. seealso::

        :func:`~matplotlib.pyplot.axes`
            For additional information on :func:`axes` and
            :func:`subplot` keyword arguments.

        :file:`gallery/pie_and_polar_charts/polar_scatter.py`
            For an example

    **Example:**

    .. plot:: gallery/subplots_axes_and_figures/subplot.py

    """
    # if subplot called without arguments, create subplot(1,1,1)
    if len(args)==0:
        args=(1,1,1)

    # This check was added because it is very easy to type
    # subplot(1, 2, False) when subplots(1, 2, False) was intended
    # (sharex=False, that is). In most cases, no error will
    # ever occur, but mysterious behavior can result because what was
    # intended to be the sharex argument is instead treated as a
    # subplot index for subplot()
    if len(args) >= 3 and isinstance(args[2], bool) :
        warnings.warn("The subplot index argument to subplot() appears"
                      " to be a boolean. Did you intend to use subplots()?")

    fig = gcf()
    a = fig.add_subplot(*args, **kwargs)
    bbox = a.bbox
    byebye = []
    for other in fig.axes:
        if other==a: continue
        if bbox.fully_overlaps(other.bbox):
            byebye.append(other)
    for ax in byebye: delaxes(ax)

    return a


def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
             subplot_kw=None, gridspec_kw=None, **fig_kw):
    """
    Create a figure and a set of subplots

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    nrows, ncols : int, optional, default: 1
        Number of rows/columns of the subplot grid.

    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (`sharex`) or y (`sharey`)
        axes:

            - True or 'all': x- or y-axis will be shared among all
              subplots.
            - False or 'none': each subplot x- or y-axis will be
              independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.

        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are visible.  Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are visible.

    squeeze : bool, optional, default: True
        - If True, extra dimensions are squeezed out from the returned Axes
          object:

            - if only one subplot is constructed (nrows=ncols=1), the
              resulting single Axes object is returned as a scalar.
            - for Nx1 or 1xN subplots, the returned object is a 1D numpy
              object array of Axes objects are returned as numpy 1D arrays.
            - for NxM, subplots with N>1 and M>1 are returned as a 2D arrays.

        - If False, no squeezing at all is done: the returned Axes object is
          always a 2D array containing Axes instances, even if it ends up
          being 1x1.

    subplot_kw : dict, optional
        Dict with keywords passed to the
        :meth:`~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.

    gridspec_kw : dict, optional
        Dict with keywords passed to the
        :class:`~matplotlib.gridspec.GridSpec` constructor used to create the
        grid the subplots are placed on.

    **fig_kw :
        All additional keyword arguments are passed to the :func:`figure` call.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object

    ax : Axes object or array of Axes objects.

        ax can be either a single :class:`matplotlib.axes.Axes` object or an
        array of Axes objects if more than one subplot was created.  The
        dimensions of the resulting array can be controlled with the squeeze
        keyword, see above.

    Examples
    --------
    First create some toy data:

    >>> x = np.linspace(0, 2*np.pi, 400)
    >>> y = np.sin(x**2)

    Creates just a figure and only one subplot

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> ax.set_title('Simple plot')

    Creates two subplots and unpacks the output array immediately

    >>> f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    >>> ax1.plot(x, y)
    >>> ax1.set_title('Sharing Y axis')
    >>> ax2.scatter(x, y)

    Creates four polar axes, and accesses them through the returned array

    >>> fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
    >>> axes[0, 0].plot(x, y)
    >>> axes[1, 1].scatter(x, y)

    Share a X axis with each column of subplots

    >>> plt.subplots(2, 2, sharex='col')

    Share a Y axis with each row of subplots

    >>> plt.subplots(2, 2, sharey='row')

    Share both X and Y axes with all subplots

    >>> plt.subplots(2, 2, sharex='all', sharey='all')

    Note that this is the same as

    >>> plt.subplots(2, 2, sharex=True, sharey=True)

    See Also
    --------
    figure
    subplot
    """
    fig = figure(**fig_kw)
    axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
                       squeeze=squeeze, subplot_kw=subplot_kw,
                       gridspec_kw=gridspec_kw)
    return fig, axs


def subplot2grid(shape, loc, rowspan=1, colspan=1, fig=None, **kwargs):
    """
    Create a subplot in a grid.  The grid is specified by *shape*, at
    location of *loc*, spanning *rowspan*, *colspan* cells in each
    direction.  The index for loc is 0-based.  The current figure will
    be used unless *fig* is specified. ::

      subplot2grid(shape, loc, rowspan=1, colspan=1)

    is identical to ::

      gridspec=GridSpec(shape[0], shape[1])
      subplotspec=gridspec.new_subplotspec(loc, rowspan, colspan)
      subplot(subplotspec)
    """

    if fig is None:
        fig = gcf()

    s1, s2 = shape
    subplotspec = GridSpec(s1, s2).new_subplotspec(loc,
                                                   rowspan=rowspan,
                                                   colspan=colspan)
    a = fig.add_subplot(subplotspec, **kwargs)
    bbox = a.bbox
    byebye = []
    for other in fig.axes:
        if other == a:
            continue
        if bbox.fully_overlaps(other.bbox):
            byebye.append(other)
    for ax in byebye:
        delaxes(ax)

    return a


def twinx(ax=None):
    """
    Make a second axes that shares the *x*-axis.  The new axes will
    overlay *ax* (or the current axes if *ax* is *None*).  The ticks
    for *ax2* will be placed on the right, and the *ax2* instance is
    returned.

    .. seealso::

       :file:`examples/api_examples/two_scales.py`
          For an example
    """
    if ax is None:
        ax = gca()
    ax1 = ax.twinx()
    return ax1


def twiny(ax=None):
    """
    Make a second axes that shares the *y*-axis.  The new axis will
    overlay *ax* (or the current axes if *ax* is *None*).  The ticks
    for *ax2* will be placed on the top, and the *ax2* instance is
    returned.
    """
    if ax is None:
        ax = gca()
    ax1 = ax.twiny()
    return ax1


def subplot_tool(targetfig=None):
    """
    Launch a subplot tool window for a figure.

    A :class:`matplotlib.widgets.SubplotTool` instance is returned.
    """
    tbar = rcParams['toolbar'] # turn off the navigation toolbar for the toolfig
    rcParams['toolbar'] = 'None'
    if targetfig is None:
        manager = get_current_fig_manager()
        targetfig = manager.canvas.figure
    else:
        # find the manager for this figure
        for manager in _pylab_helpers.Gcf._activeQue:
            if manager.canvas.figure==targetfig: break
        else: raise RuntimeError('Could not find manager for targetfig')

    toolfig = figure(figsize=(6,3))
    toolfig.subplots_adjust(top=0.9)
    ret =  SubplotTool(targetfig, toolfig)
    rcParams['toolbar'] = tbar
    _pylab_helpers.Gcf.set_active(manager)  # restore the current figure
    return ret


def box(on=None):
    """
    Turn the axes box on or off.  *on* may be a boolean or a string,
    'on' or 'off'.

    If *on* is *None*, toggle state.
    """
    ax = gca()
    on = _string_to_bool(on)
    if on is None:
        on = not ax.get_frame_on()
    ax.set_frame_on(on)


## Axis ##


def xlim(*args, **kwargs):
    """
    Get or set the *x* limits of the current axes.

    ::

      xmin, xmax = xlim()   # return the current xlim
      xlim( (xmin, xmax) )  # set the xlim to xmin, xmax
      xlim( xmin, xmax )    # set the xlim to xmin, xmax

    If you do not specify args, you can pass the xmin and xmax as
    kwargs, e.g.::

      xlim(xmax=3) # adjust the max leaving min unchanged
      xlim(xmin=1) # adjust the min leaving max unchanged

    Setting limits turns autoscaling off for the x-axis.

    The new axis limits are returned as a length 2 tuple.

    """
    ax = gca()
    if not args and not kwargs:
        return ax.get_xlim()
    ret = ax.set_xlim(*args, **kwargs)
    return ret


def ylim(*args, **kwargs):
    """
    Get or set the *y*-limits of the current axes.

    ::

      ymin, ymax = ylim()   # return the current ylim
      ylim( (ymin, ymax) )  # set the ylim to ymin, ymax
      ylim( ymin, ymax )    # set the ylim to ymin, ymax

    If you do not specify args, you can pass the *ymin* and *ymax* as
    kwargs, e.g.::

      ylim(ymax=3) # adjust the max leaving min unchanged
      ylim(ymin=1) # adjust the min leaving max unchanged

    Setting limits turns autoscaling off for the y-axis.

    The new axis limits are returned as a length 2 tuple.
    """
    ax = gca()
    if not args and not kwargs:
        return ax.get_ylim()
    ret = ax.set_ylim(*args, **kwargs)
    return ret


def xticks(*args, **kwargs):
    """
    Get or set the *x*-limits of the current tick locations and labels.

    ::

      # return locs, labels where locs is an array of tick locations and
      # labels is an array of tick labels.
      locs, labels = xticks()

      # set the locations of the xticks
      xticks( arange(6) )

      # set the locations and labels of the xticks
      xticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )

    The keyword args, if any, are :class:`~matplotlib.text.Text`
    properties. For example, to rotate long labels::

      xticks( arange(12), calendar.month_name[1:13], rotation=17 )
    """
    ax = gca()
    if len(args) == 0:
        locs = ax.get_xticks()
        labels = ax.get_xticklabels()
    elif len(args) == 1:
        locs = ax.set_xticks(args[0])
        labels = ax.get_xticklabels()
    elif len(args) == 2:
        locs = ax.set_xticks(args[0])
        labels = ax.set_xticklabels(args[1], **kwargs)
    else:
        raise TypeError('Illegal number of arguments to xticks')
    for l in labels:
        l.update(kwargs)
    return locs, silent_list('Text xticklabel', labels)


def yticks(*args, **kwargs):
    """
    Get or set the *y*-limits of the current tick locations and labels.

    ::

      # return locs, labels where locs is an array of tick locations and
      # labels is an array of tick labels.
      locs, labels = yticks()

      # set the locations of the yticks
      yticks( arange(6) )

      # set the locations and labels of the yticks
      yticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )

    The keyword args, if any, are :class:`~matplotlib.text.Text`
    properties. For example, to rotate long labels::

      yticks( arange(12), calendar.month_name[1:13], rotation=45 )
    """
    ax = gca()
    if len(args) == 0:
        locs = ax.get_yticks()
        labels = ax.get_yticklabels()
    elif len(args) == 1:
        locs = ax.set_yticks(args[0])
        labels = ax.get_yticklabels()
    elif len(args) == 2:
        locs = ax.set_yticks(args[0])
        labels = ax.set_yticklabels(args[1], **kwargs)
    else:
        raise TypeError('Illegal number of arguments to yticks')
    for l in labels:
        l.update(kwargs)
    return locs, silent_list('Text yticklabel', labels)


def rgrids(*args, **kwargs):
    """
    Get or set the radial gridlines on a polar plot.

    call signatures::

      lines, labels = rgrids()
      lines, labels = rgrids(radii, labels=None, angle=22.5, **kwargs)

    When called with no arguments, :func:`rgrid` simply returns the
    tuple (*lines*, *labels*), where *lines* is an array of radial
    gridlines (:class:`~matplotlib.lines.Line2D` instances) and
    *labels* is an array of tick labels
    (:class:`~matplotlib.text.Text` instances). When called with
    arguments, the labels will appear at the specified radial
    distances and angles.

    *labels*, if not *None*, is a len(*radii*) list of strings of the
    labels to use at each angle.

    If *labels* is None, the rformatter will be used

    Examples::

      # set the locations of the radial gridlines and labels
      lines, labels = rgrids( (0.25, 0.5, 1.0) )

      # set the locations and labels of the radial gridlines and labels
      lines, labels = rgrids( (0.25, 0.5, 1.0), ('Tom', 'Dick', 'Harry' )

    """
    ax = gca()
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('rgrids only defined for polar axes')
    if len(args)==0:
        lines = ax.yaxis.get_gridlines()
        labels = ax.yaxis.get_ticklabels()
    else:
        lines, labels = ax.set_rgrids(*args, **kwargs)

    return ( silent_list('Line2D rgridline', lines),
             silent_list('Text rgridlabel', labels) )


def thetagrids(*args, **kwargs):
    """
    Get or set the theta locations of the gridlines in a polar plot.

    If no arguments are passed, return a tuple (*lines*, *labels*)
    where *lines* is an array of radial gridlines
    (:class:`~matplotlib.lines.Line2D` instances) and *labels* is an
    array of tick labels (:class:`~matplotlib.text.Text` instances)::

      lines, labels = thetagrids()

    Otherwise the syntax is::

      lines, labels = thetagrids(angles, labels=None, fmt='%d', frac = 1.1)

    set the angles at which to place the theta grids (these gridlines
    are equal along the theta dimension).

    *angles* is in degrees.

    *labels*, if not *None*, is a len(angles) list of strings of the
    labels to use at each angle.

    If *labels* is *None*, the labels will be ``fmt%angle``.

    *frac* is the fraction of the polar axes radius at which to place
    the label (1 is the edge). e.g., 1.05 is outside the axes and 0.95
    is inside the axes.

    Return value is a list of tuples (*lines*, *labels*):

      - *lines* are :class:`~matplotlib.lines.Line2D` instances

      - *labels* are :class:`~matplotlib.text.Text` instances.

    Note that on input, the *labels* argument is a list of strings,
    and on output it is a list of :class:`~matplotlib.text.Text`
    instances.

    Examples::

      # set the locations of the radial gridlines and labels
      lines, labels = thetagrids( range(45,360,90) )

      # set the locations and labels of the radial gridlines and labels
      lines, labels = thetagrids( range(45,360,90), ('NE', 'NW', 'SW','SE') )
    """
    ax = gca()
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('rgrids only defined for polar axes')
    if len(args)==0:
        lines = ax.xaxis.get_ticklines()
        labels = ax.xaxis.get_ticklabels()
    else:
        lines, labels = ax.set_thetagrids(*args, **kwargs)

    return (silent_list('Line2D thetagridline', lines),
            silent_list('Text thetagridlabel', labels)
            )


## Plotting Info ##

def plotting():
    pass


@deprecated('2.2')
def get_plot_commands():
    """Get a sorted list of all of the plotting commands."""
    # This works by searching for all functions in this module and
    # removing a few hard-coded exclusions, as well as all of the
    # colormap-setting functions, and anything marked as private with
    # a preceding underscore.

    exclude = {'colormaps', 'colors', 'connect', 'disconnect',
               'get_plot_commands', 'get_current_fig_manager', 'ginput',
               'plotting', 'waitforbuttonpress'}
    return sorted(set(__all__) - exclude - set(colormaps()))


@deprecated('2.1')
def colors():
    """
    This is a do-nothing function to provide you with help on how
    matplotlib handles colors.

    Commands which take color arguments can use several formats to
    specify the colors.  For the basic built-in colors, you can use a
    single letter

      =====   =======
      Alias   Color
      =====   =======
      'b'     blue
      'g'     green
      'r'     red
      'c'     cyan
      'm'     magenta
      'y'     yellow
      'k'     black
      'w'     white
      =====   =======

    For a greater range of colors, you have two options.  You can
    specify the color using an html hex string, as in::

      color = '#eeefff'

    or you can pass an R,G,B tuple, where each of R,G,B are in the
    range [0,1].

    You can also use any legal html name for a color, for example::

      color = 'red'
      color = 'burlywood'
      color = 'chartreuse'

    The example below creates a subplot with a dark
    slate gray background::

       subplot(111, facecolor=(0.1843, 0.3098, 0.3098))

    Here is an example that creates a pale turquoise title::

      title('Is this the best color?', color='#afeeee')

    """
    pass


def colormaps():
    """
    Matplotlib provides a number of colormaps, and others can be added using
    :func:`~matplotlib.cm.register_cmap`.  This function documents the built-in
    colormaps, and will also return a list of all registered colormaps if called.

    You can set the colormap for an image, pcolor, scatter, etc,
    using a keyword argument::

      imshow(X, cmap=cm.hot)

    or using the :func:`set_cmap` function::

      imshow(X)
      pyplot.set_cmap('hot')
      pyplot.set_cmap('jet')

    In interactive mode, :func:`set_cmap` will update the colormap post-hoc,
    allowing you to see which one works best for your data.

    All built-in colormaps can be reversed by appending ``_r``: For instance,
    ``gray_r`` is the reverse of ``gray``.

    There are several common color schemes used in visualization:

    Sequential schemes
      for unipolar data that progresses from low to high
    Diverging schemes
      for bipolar data that emphasizes positive or negative deviations from a
      central value
    Cyclic schemes
      meant for plotting values that wrap around at the
      endpoints, such as phase angle, wind direction, or time of day
    Qualitative schemes
      for nominal data that has no inherent ordering, where color is used
      only to distinguish categories

    Matplotlib ships with 4 perceptually uniform color maps which are
    the recommended color maps for sequential data:

      =========   ===================================================
      Colormap    Description
      =========   ===================================================
      inferno     perceptually uniform shades of black-red-yellow
      magma       perceptually uniform shades of black-red-white
      plasma      perceptually uniform shades of blue-red-yellow
      viridis     perceptually uniform shades of blue-green-yellow
      =========   ===================================================

    The following colormaps are based on the `ColorBrewer
    <http://colorbrewer2.org>`_ color specifications and designs developed by
    Cynthia Brewer:

    ColorBrewer Diverging (luminance is highest at the midpoint, and
    decreases towards differently-colored endpoints):

      ========  ===================================
      Colormap  Description
      ========  ===================================
      BrBG      brown, white, blue-green
      PiYG      pink, white, yellow-green
      PRGn      purple, white, green
      PuOr      orange, white, purple
      RdBu      red, white, blue
      RdGy      red, white, gray
      RdYlBu    red, yellow, blue
      RdYlGn    red, yellow, green
      Spectral  red, orange, yellow, green, blue
      ========  ===================================

    ColorBrewer Sequential (luminance decreases monotonically):

      ========  ====================================
      Colormap  Description
      ========  ====================================
      Blues     white to dark blue
      BuGn      white, light blue, dark green
      BuPu      white, light blue, dark purple
      GnBu      white, light green, dark blue
      Greens    white to dark green
      Greys     white to black (not linear)
      Oranges   white, orange, dark brown
      OrRd      white, orange, dark red
      PuBu      white, light purple, dark blue
      PuBuGn    white, light purple, dark green
      PuRd      white, light purple, dark red
      Purples   white to dark purple
      RdPu      white, pink, dark purple
      Reds      white to dark red
      YlGn      light yellow, dark green
      YlGnBu    light yellow, light green, dark blue
      YlOrBr    light yellow, orange, dark brown
      YlOrRd    light yellow, orange, dark red
      ========  ====================================

    ColorBrewer Qualitative:

    (For plotting nominal data, :class:`ListedColormap` is used,
    not :class:`LinearSegmentedColormap`.  Different sets of colors are
    recommended for different numbers of categories.)

    * Accent
    * Dark2
    * Paired
    * Pastel1
    * Pastel2
    * Set1
    * Set2
    * Set3

    A set of colormaps derived from those of the same name provided
    with Matlab are also included:

      =========   =======================================================
      Colormap    Description
      =========   =======================================================
      autumn      sequential linearly-increasing shades of red-orange-yellow
      bone        sequential increasing black-white color map with
                  a tinge of blue, to emulate X-ray film
      cool        linearly-decreasing shades of cyan-magenta
      copper      sequential increasing shades of black-copper
      flag        repetitive red-white-blue-black pattern (not cyclic at
                  endpoints)
      gray        sequential linearly-increasing black-to-white
                  grayscale
      hot         sequential black-red-yellow-white, to emulate blackbody
                  radiation from an object at increasing temperatures
      hsv         cyclic red-yellow-green-cyan-blue-magenta-red, formed
                  by changing the hue component in the HSV color space
      jet         a spectral map with dark endpoints, blue-cyan-yellow-red;
                  based on a fluid-jet simulation by NCSA [#]_
      pink        sequential increasing pastel black-pink-white, meant
                  for sepia tone colorization of photographs
      prism       repetitive red-yellow-green-blue-purple-...-green pattern
                  (not cyclic at endpoints)
      spring      linearly-increasing shades of magenta-yellow
      summer      sequential linearly-increasing shades of green-yellow
      winter      linearly-increasing shades of blue-green
      =========   =======================================================

    A set of palettes from the `Yorick scientific visualisation
    package <https://dhmunro.github.io/yorick-doc/>`_, an evolution of
    the GIST package, both by David H. Munro are included:

      ============  =======================================================
      Colormap      Description
      ============  =======================================================
      gist_earth    mapmaker's colors from dark blue deep ocean to green
                    lowlands to brown highlands to white mountains
      gist_heat     sequential increasing black-red-orange-white, to emulate
                    blackbody radiation from an iron bar as it grows hotter
      gist_ncar     pseudo-spectral black-blue-green-yellow-red-purple-white
                    colormap from National Center for Atmospheric
                    Research [#]_
      gist_rainbow  runs through the colors in spectral order from red to
                    violet at full saturation (like *hsv* but not cyclic)
      gist_stern    "Stern special" color table from Interactive Data
                    Language software
      ============  =======================================================


    Other miscellaneous schemes:

      ============= =======================================================
      Colormap      Description
      ============= =======================================================
      afmhot        sequential black-orange-yellow-white blackbody
                    spectrum, commonly used in atomic force microscopy
      brg           blue-red-green
      bwr           diverging blue-white-red
      coolwarm      diverging blue-gray-red, meant to avoid issues with 3D
                    shading, color blindness, and ordering of colors [#]_
      CMRmap        "Default colormaps on color images often reproduce to
                    confusing grayscale images. The proposed colormap
                    maintains an aesthetically pleasing color image that
                    automatically reproduces to a monotonic grayscale with
                    discrete, quantifiable saturation levels." [#]_
      cubehelix     Unlike most other color schemes cubehelix was designed
                    by D.A. Green to be monotonically increasing in terms
                    of perceived brightness. Also, when printed on a black
                    and white postscript printer, the scheme results in a
                    greyscale with monotonically increasing brightness.
                    This color scheme is named cubehelix because the r,g,b
                    values produced can be visualised as a squashed helix
                    around the diagonal in the r,g,b color cube.
      gnuplot       gnuplot's traditional pm3d scheme
                    (black-blue-red-yellow)
      gnuplot2      sequential color printable as gray
                    (black-blue-violet-yellow-white)
      ocean         green-blue-white
      rainbow       spectral purple-blue-green-yellow-orange-red colormap
                    with diverging luminance
      seismic       diverging blue-white-red
      nipy_spectral black-purple-blue-green-yellow-red-white spectrum,
                    originally from the Neuroimaging in Python project
      terrain       mapmaker's colors, blue-green-yellow-brown-white,
                    originally from IGOR Pro
      ============= =======================================================

    The following colormaps are redundant and may be removed in future
    versions.  It's recommended to use the names in the descriptions
    instead, which produce identical output:

      =========  =======================================================
      Colormap   Description
      =========  =======================================================
      gist_gray  identical to *gray*
      gist_yarg  identical to *gray_r*
      binary     identical to *gray_r*
      spectral   identical to *nipy_spectral* [#]_
      =========  =======================================================

    .. rubric:: Footnotes

    .. [#] Rainbow colormaps, ``jet`` in particular, are considered a poor
      choice for scientific visualization by many researchers: `Rainbow Color
      Map (Still) Considered Harmful
      <http://ieeexplore.ieee.org/document/4118486/?arnumber=4118486>`_

    .. [#] Resembles "BkBlAqGrYeOrReViWh200" from NCAR Command
      Language. See `Color Table Gallery
      <https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml>`_

    .. [#] See `Diverging Color Maps for Scientific Visualization
      <http://www.kennethmoreland.com/color-maps/>`_ by Kenneth Moreland.

    .. [#] See `A Color Map for Effective Black-and-White Rendering of
      Color-Scale Images
      <https://www.mathworks.com/matlabcentral/fileexchange/2662-cmrmap-m>`_
      by Carey Rappaport

    .. [#] Changed to distinguish from ColorBrewer's *Spectral* map.
      :func:`spectral` still works, but
      ``set_cmap('nipy_spectral')`` is recommended for clarity.


    """
    return sorted(cm.cmap_d)


def _setup_pyplot_info_docstrings():
    """
    Generates the plotting and docstring.

    These must be done after the entire module is imported, so it is called
    from the end of this module.
    """
    # Generate the plotting docstring

    def pad(s, l):
        """Pad string *s* to length *l*."""
        if l < len(s):
            return s[:l]
        return s + ' ' * (l - len(s))

    # Searching for all public functions in this module (this is done when
    # defining __all__) and remove a few hard-coded exclusions, as well as all
    # of the colormap-setting functions
    exclude = {"colormaps", "colors", "connect", "disconnect",
               "get_current_fig_manager", "ginput", "plotting",
               "waitforbuttonpress"}
    commands =  sorted(set(__all__) - exclude - set(colormaps()))

    first_sentence = re.compile(r"(?:\s*).+?\.(?:\s+|$)", flags=re.DOTALL)

    # Collect the first sentence of the docstring for all of the
    # plotting commands.
    rows = []
    max_name = 0
    max_summary = 0
    for name in commands:
        doc = globals()[name].__doc__
        summary = ''
        if doc is not None:
            match = first_sentence.match(doc)
            if match is not None:
                summary = match.group(0).strip().replace('\n', ' ')
        name = '`%s`' % name
        rows.append([name, summary])
        max_name = max(max_name, len(name))
        max_summary = max(max_summary, len(summary))

    lines = []
    sep = '=' * max_name + ' ' + '=' * max_summary
    lines.append(sep)
    lines.append(' '.join([pad("Function", max_name),
                           pad("Description", max_summary)]))
    lines.append(sep)
    for name, summary in rows:
        lines.append(' '.join([pad(name, max_name),
                               pad(summary, max_summary)]))
    lines.append(sep)

    plotting.__doc__ = '\n'.join(lines)

## Plotting part 1: manually generated functions and wrappers ##

def colorbar(mappable=None, cax=None, ax=None, **kw):
    if mappable is None:
        mappable = gci()
        if mappable is None:
            raise RuntimeError('No mappable was found to use for colorbar '
                               'creation. First define a mappable such as '
                               'an image (with imshow) or a contour set ('
                               'with contourf).')
    if ax is None:
        ax = gca()

    ret = gcf().colorbar(mappable, cax = cax, ax=ax, **kw)
    return ret
colorbar.__doc__ = matplotlib.colorbar.colorbar_doc


def clim(vmin=None, vmax=None):
    """
    Set the color limits of the current image.

    To apply clim to all axes images do::

      clim(0, 0.5)

    If either *vmin* or *vmax* is None, the image min/max respectively
    will be used for color scaling.

    If you want to set the clim of multiple images,
    use, for example::

      for im in gca().get_images():
          im.set_clim(0, 0.05)

    """
    im = gci()
    if im is None:
        raise RuntimeError('You must first define an image, e.g., with imshow')

    im.set_clim(vmin, vmax)


def set_cmap(cmap):
    """
    Set the default colormap.  Applies to the current image if any.
    See help(colormaps) for more information.

    *cmap* must be a :class:`~matplotlib.colors.Colormap` instance, or
    the name of a registered colormap.

    See :func:`matplotlib.cm.register_cmap` and
    :func:`matplotlib.cm.get_cmap`.
    """
    cmap = cm.get_cmap(cmap)

    rc('image', cmap=cmap.name)
    im = gci()

    if im is not None:
        im.set_cmap(cmap)


def matshow(A, fignum=None, **kw):
    """
    Display an array as a matrix in a new figure window.

    The origin is set at the upper left hand corner and rows (first
    dimension of the array) are displayed horizontally.  The aspect
    ratio of the figure window is that of the array, unless this would
    make an excessively short or narrow figure.

    Tick labels for the xaxis are placed on top.

    With the exception of *fignum*, keyword arguments are passed to
    :func:`~matplotlib.pyplot.imshow`.  You may set the *origin*
    kwarg to "lower" if you want the first row in the array to be
    at the bottom instead of the top.


    *fignum*: [ None | integer | False ]
      By default, :func:`matshow` creates a new figure window with
      automatic numbering.  If *fignum* is given as an integer, the
      created figure will use this figure number.  Because of how
      :func:`matshow` tries to set the figure aspect ratio to be the
      one of the array, if you provide the number of an already
      existing figure, strange things may happen.

      If *fignum* is *False* or 0, a new figure window will **NOT** be created.
    """
    A = np.asanyarray(A)
    if fignum is False or fignum is 0:
        ax = gca()
    else:
        # Extract actual aspect ratio of array and make appropriately sized figure
        fig = figure(fignum, figsize=figaspect(A))
        ax  = fig.add_axes([0.15, 0.09, 0.775, 0.775])

    im = ax.matshow(A, **kw)
    sci(im)

    return im


def polar(*args, **kwargs):
    """
    Make a polar plot.

    call signature::

      polar(theta, r, **kwargs)

    Multiple *theta*, *r* arguments are supported, with format
    strings, as in :func:`~matplotlib.pyplot.plot`.

    """
    # If an axis already exists, check if it has a polar projection
    if gcf().get_axes():
        if not isinstance(gca(), PolarAxes):
            warnings.warn('Trying to create polar plot on an axis that does '
                          'not have a polar projection.')
    ax = gca(polar=True)
    ret = ax.plot(*args, **kwargs)
    return ret


def plotfile(fname, cols=(0,), plotfuncs=None,
             comments='#', skiprows=0, checkrows=5, delimiter=',',
             names=None, subplots=True, newfig=True, **kwargs):
    """
    Plot the data in a file.

    *cols* is a sequence of column identifiers to plot.  An identifier
    is either an int or a string.  If it is an int, it indicates the
    column number.  If it is a string, it indicates the column header.
    matplotlib will make column headers lower case, replace spaces with
    underscores, and remove all illegal characters; so ``'Adj Close*'``
    will have name ``'adj_close'``.

    - If len(*cols*) == 1, only that column will be plotted on the *y* axis.

    - If len(*cols*) > 1, the first element will be an identifier for
      data for the *x* axis and the remaining elements will be the
      column indexes for multiple subplots if *subplots* is *True*
      (the default), or for lines in a single subplot if *subplots*
      is *False*.

    *plotfuncs*, if not *None*, is a dictionary mapping identifier to
    an :class:`~matplotlib.axes.Axes` plotting function as a string.
    Default is 'plot', other choices are 'semilogy', 'fill', 'bar',
    etc.  You must use the same type of identifier in the *cols*
    vector as you use in the *plotfuncs* dictionary, e.g., integer
    column numbers in both or column names in both. If *subplots*
    is *False*, then including any function such as 'semilogy'
    that changes the axis scaling will set the scaling for all
    columns.

    *comments*, *skiprows*, *checkrows*, *delimiter*, and *names*
    are all passed on to :func:`matplotlib.pylab.csv2rec` to
    load the data into a record array.

    If *newfig* is *True*, the plot always will be made in a new figure;
    if *False*, it will be made in the current figure if one exists,
    else in a new figure.

    kwargs are passed on to plotting functions.

    Example usage::

      # plot the 2nd and 4th column against the 1st in two subplots
      plotfile(fname, (0,1,3))

      # plot using column names; specify an alternate plot type for volume
      plotfile(fname, ('date', 'volume', 'adj_close'),
                                    plotfuncs={'volume': 'semilogy'})

    Note: plotfile is intended as a convenience for quickly plotting
    data from flat files; it is not intended as an alternative
    interface to general plotting with pyplot or matplotlib.
    """

    if newfig:
        fig = figure()
    else:
        fig = gcf()

    if len(cols) < 1:
        raise ValueError('must have at least one column of data')

    if plotfuncs is None:
        plotfuncs = {}
    r = mlab.csv2rec(fname, comments=comments, skiprows=skiprows,
                     checkrows=checkrows, delimiter=delimiter, names=names)

    def getname_val(identifier):
        'return the name and column data for identifier'
        if isinstance(identifier, six.string_types):
            return identifier, r[identifier]
        elif is_numlike(identifier):
            name = r.dtype.names[int(identifier)]
            return name, r[name]
        else:
            raise TypeError('identifier must be a string or integer')

    xname, x = getname_val(cols[0])
    ynamelist = []

    if len(cols) == 1:
        ax1 = fig.add_subplot(1, 1, 1)
        funcname = plotfuncs.get(cols[0], 'plot')
        func = getattr(ax1, funcname)
        func(x, **kwargs)
        ax1.set_ylabel(xname)
    else:
        N = len(cols)
        for i in range(1, N):
            if subplots:
                if i == 1:
                    ax = ax1 = fig.add_subplot(N - 1, 1, i)
                else:
                    ax = fig.add_subplot(N - 1, 1, i, sharex=ax1)
            elif i == 1:
                ax = fig.add_subplot(1, 1, 1)

            yname, y = getname_val(cols[i])
            ynamelist.append(yname)

            funcname = plotfuncs.get(cols[i], 'plot')
            func = getattr(ax, funcname)

            func(x, y, **kwargs)
            if subplots:
                ax.set_ylabel(yname)
            if ax.is_last_row():
                ax.set_xlabel(xname)
            else:
                ax.set_xlabel('')

    if not subplots:
        ax.legend(ynamelist, loc='best')

    if xname == 'date':
        fig.autofmt_xdate()


# just to be safe.  Interactive mode can be turned on without
# calling `plt.ion()` so register it again here.
# This is safe because multiple calls to `install_repl_displayhook`
# are no-ops and the registered function respect `mpl.is_interactive()`
# to determine if they should trigger a draw.
install_repl_displayhook()


def _make_wrapper(obj_getter, cls, command):
    if ":" in command:
        method_name, func_name = command.split(":")
    else:
        method_name = func_name = command
    method = getattr(cls, method_name)
    post_processor = _post_processors.get(func_name, lambda obj: None)

    def func(*args, **kwargs):
        # We actually need to refetch the method here in case it has been
        # monkey-patched.
        ret = getattr(obj_getter(), method_name)(*args, **kwargs)
        post_processor(ret)
        return ret

    if six.PY2:
        func.__name__ = func_name.encode("ascii")
        import funcsigs
        sig = funcsigs.signature(method)
    else:
        func.__name__ = func_name
        sig = inspect.signature(method)

    func.__doc__ = method.__doc__
    func.__signature__ = sig.replace(
        parameters=list(sig.parameters.values())[1:])
    globals()[func.__name__] = func


_canvas_commands = ["draw", "mpl_connect:connect", "mpl_disconnect:disconnect"]
_fig_commands = [
    "clf", "figimage", "gca", "ginput", "savefig", "subplots_adjust",
    "suptitle", "tight_layout", "waitforbuttonpress",
    # Renamed commands.
    "_gci:gci", "legend:figlegend", "text:figtext"]
_axes_commands = [
    "acorr", "angle_spectrum", "annotate", "arrow", "autoscale", "axhline",
    "axhspan", "axis", "axvline", "axvspan", "bar", "barbs", "barh", "boxplot",
    "broken_barh", "cla", "clabel", "cohere", "contour", "contourf", "csd",
    "errorbar", "eventplot", "fill", "fill_between", "fill_betweenx", "grid",
    "hexbin", "hist", "hist2d", "hlines", "imshow", "legend", "locator_params",
    "loglog", "magnitude_spectrum", "margins", "minorticks_off",
    "minorticks_on", "pcolor", "pcolormesh", "phase_spectrum", "pie", "plot",
    "plot_date", "psd", "quiver", "quiverkey", "scatter", "semilogx",
    "semilogy", "specgram", "stackplot", "stem", "step", "streamplot", "table",
    "text", "tick_params", "ticklabel_format", "tricontour", "tricontourf",
    "tripcolor", "triplot", "violinplot", "vlines", "xcorr",
    # Renamed commands.
    "set_title:title", "set_xlabel:xlabel", "set_xscale:xscale",
    "set_ylabel:ylabel", "set_yscale:yscale"]
_post_processors = {
    # For the following functions, an additional callable will be called with
    # the return value as argument.
    "hexbin": sci,
    "imshow": sci,
    "pcolor": sci,
    "pcolormesh": sci,
    "quiver": sci,
    "scatter": sci,
    "tripcolor": sci,
    "contour": lambda ret: sci(ret) if ret._A is not None else None,
    "contourf": lambda ret: sci(ret) if ret._A is not None else None,
    "tricontour": lambda ret: sci(ret) if ret._A is not None else None,
    "tricontourf": lambda ret: sci(ret) if ret._A is not None else None,
    "hist2d": lambda ret: sci(ret[-1]),
    "specgram": lambda ret: sci(ret[-1]),
    "streamplot": lambda ret: sci(ret.lines),
    "spy": (
        lambda ret: sci(ret) if isinstance(ret, cm.ScalarMappable) else None)
}

for _command in _canvas_commands:
    _make_wrapper(
        lambda: get_current_fig_manager().canvas, FigureCanvasBase, _command)
for _command in _fig_commands:
    _make_wrapper(gcf, Figure, _command)
for _command in _axes_commands:
    _make_wrapper(gca, Axes, _command)


def _make_cmap_wrapper(name):

    def func():
        rc("image", cmap=name)
        im = gci()
        if im is not None:
            im.set_cmap(name)

    func.__name__ = name.encode("ascii") if six.PY2 else name
    func.__doc__ = """
    Set the default colormap to "{}" and apply it to the current image, if any.

    See ``help(colormaps)`` for more information.
    """.format(name)
    globals()[func.__name__] = func


_cmaps = [
    "autumn", "bone", "cool", "copper", "flag", "gray", "hot", "hsv",
    "inferno", "jet", "magma", "nipy_spectral", "pink", "plasma", "prism",
    "spectral", "spring", "summer", "viridis", "winter"]


for _cmap in _cmaps:
    _make_cmap_wrapper(_cmap)


__all__ = sorted([name for name, obj in globals().items()
                  if getattr(obj, "__module__", None) == __name__
                     and not name.startswith("_")]
                 + ["getp", "imread", "imsave", "rc", "rc_context", "setp"])


_setup_pyplot_info_docstrings()
