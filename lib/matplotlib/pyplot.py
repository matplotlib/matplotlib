import sys

import matplotlib
from matplotlib import _pylab_helpers, interactive
from matplotlib.cbook import dedent, silent_list, is_string_like, is_numlike
from matplotlib.figure import Figure, figaspect
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.image import imread as _imread
from matplotlib.image import imsave as _imsave
from matplotlib import rcParams, rcParamsDefault, get_backend
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import getp, get, Artist
from matplotlib.artist import setp as _setp
from matplotlib.axes import Axes
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for csv2rec, detrend_none, window_hanning
from matplotlib.scale import get_scale_docs, get_scale_names

from matplotlib import cm
from matplotlib.cm import get_cmap

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize, normalize # latter for backwards compat.
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import SubplotTool, Button, Slider, Widget

from ticker import TickHelper, Formatter, FixedFormatter, NullFormatter,\
           FuncFormatter, FormatStrFormatter, ScalarFormatter,\
           LogFormatter, LogFormatterExponent, LogFormatterMathtext,\
           Locator, IndexLocator, FixedLocator, NullLocator,\
           LinearLocator, LogLocator, AutoLocator, MultipleLocator,\
           MaxNLocator


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
    elif 'qt' in sys.modules and not backend == 'QtAgg':
        import qt
        if not qt.qApp.startingUp():
            # The mainloop is running.
            rcParams['backend'] = 'qtAgg'
    elif 'PyQt4.QtCore' in sys.modules and not backend == 'Qt4Agg':
        import PyQt4.QtGui
        if not PyQt4.QtGui.qApp.startingUp():
            # The mainloop is running.
            rcParams['backend'] = 'qt4Agg'
    elif 'gtk' in sys.modules and not backend in ('GTK', 'GTKAgg',
                                                            'GTKCairo'):
        import gobject
        if gobject.MainLoop().is_running():
            rcParams['backend'] = 'gtk' + 'Agg' * is_agg_backend
    elif 'Tkinter' in sys.modules and not backend == 'TkAgg':
        #import Tkinter
        pass #what if anything do we need to do for tkinter?

_backend_selection()

## Global ##

from matplotlib.backends import pylab_setup
new_figure_manager, draw_if_interactive, show = pylab_setup()



def findobj(o=None, match=None):
    if o is None:
        o = gcf()
    return o.findobj(match)
findobj.__doc__ = Artist.findobj.__doc__

def switch_backend(newbackend):
    """
    Switch the default backend to newbackend.  This feature is
    **experimental**, and is only expected to work switching to an
    image backend.  Eg, if you have a bunch of PostScript scripts that
    you want to run from an interactive ipython session, you may want
    to switch to the PS backend before running them to avoid having a
    bunch of GUI windows popup.  If you try to interactively switch
    from one GUI backend to another, you will explode.

    Calling this command will close all open windows.
    """
    close('all')
    global new_figure_manager, draw_if_interactive, show
    matplotlib.use(newbackend, warn=False)
    reload(matplotlib.backends)
    from matplotlib.backends import pylab_setup
    new_figure_manager, draw_if_interactive, show = pylab_setup()


def isinteractive():
    """
    Return the interactive status
    """
    return matplotlib.is_interactive()

def ioff():
    'Turn interactive mode off.'
    matplotlib.interactive(False)

def ion():
    'Turn interactive mode on.'
    matplotlib.interactive(True)

def rc(*args, **kwargs):
    matplotlib.rc(*args, **kwargs)
if matplotlib.rc.__doc__ is not None:
    rc.__doc__ =  dedent(matplotlib.rc.__doc__)

def rcdefaults():
    matplotlib.rcdefaults()
    draw_if_interactive()
if matplotlib.rcdefaults.__doc__ is not None:
    rcdefaults.__doc__ =   dedent(matplotlib.rcdefaults.__doc__)

# The current "image" (ScalarMappable) is tracked here on a
# per-pylab-session basis:
def gci():
    """
    Get the current :class:`~matplotlib.cm.ScalarMappable` instance
    (image or patch collection), or *None* if no images or patch
    collections have been defined.  The commands
    :func:`~matplotlib.pyplot.imshow` and
    :func:`~matplotlib.pyplot.figimage` create
    :class:`~matplotlib.image.Image` instances, and the commands
    :func:`~matplotlib.pyplot.pcolor` and
    :func:`~matplotlib.pyplot.scatter` create
    :class:`~matplotlib.collections.Collection` instances.
    """
    return gci._current
gci._current = None


def sci(im):
    """
    Set the current image (target of colormap commands like
    :func:`~matplotlib.pyplot.jet`, :func:`~matplotlib.pyplot.hot` or
    :func:`~matplotlib.pyplot.clim`).
    """
    gci._current = im


## Any Artist ##

# (getp is simply imported)

def setp(*args, **kwargs):
    ret = _setp(*args, **kwargs)
    draw_if_interactive()
    return ret
if _setp.__doc__ is not None:
    setp.__doc__ = _setp.__doc__




## Figures ##



def figure(num=None, # autoincrement if None, else integer from 1-N
           figsize   = None, # defaults to rc figure.figsize
           dpi       = None, # defaults to rc figure.dpi
           facecolor = None, # defaults to rc figure.facecolor
           edgecolor = None, # defaults to rc figure.edgecolor
           frameon = True,
           FigureClass = Figure,
           **kwargs
           ):
    """
    call signature::

      figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


    Create a new figure and return a :class:`matplotlib.figure.Figure`
    instance.  If *num* = *None*, the figure number will be incremented and
    a new figure will be created.  The returned figure objects have a
    *number* attribute holding this number.

    If *num* is an integer, and ``figure(num)`` already exists, make it
    active and return a reference to it.  If ``figure(num)`` does not exist
    it will be created.  Numbering starts at 1, matlab style::

      figure(1)

    If you are creating many figures, make sure you explicitly call "close"
    on the figures you are not using, because this will enable pylab
    to properly clean up the memory.

    Optional keyword arguments:

      =========   =======================================================
      Keyword     Description
      =========   =======================================================
      figsize     width x height in inches; defaults to rc figure.figsize
      dpi         resolution; defaults to rc figure.dpi
      facecolor   the background color; defaults to rc figure.facecolor
      edgecolor   the border color; defaults to rc figure.edgecolor
      =========   =======================================================

    rcParams defines the default values, which can be modified in the
    matplotlibrc file

    *FigureClass* is a :class:`~matplotlib.figure.Figure` or derived
    class that will be passed on to :meth:`new_figure_manager` in the
    backends which allows you to hook custom Figure classes into the
    pylab interface.  Additional kwargs will be passed on to your
    figure init function.
    """

    if figsize is None   : figsize   = rcParams['figure.figsize']
    if dpi is None       : dpi       = rcParams['figure.dpi']
    if facecolor is None : facecolor = rcParams['figure.facecolor']
    if edgecolor is None : edgecolor = rcParams['figure.edgecolor']

    if num is None:
        allnums = [f.num for f in _pylab_helpers.Gcf.get_all_fig_managers()]
        if allnums:
            num = max(allnums) + 1
        else:
            num = 1
    else:
        num = int(num)  # crude validation of num argument


    figManager = _pylab_helpers.Gcf.get_fig_manager(num)
    if figManager is None:
        if get_backend().lower() == 'ps':  dpi = 72

        figManager = new_figure_manager(num, figsize=figsize,
                                             dpi=dpi,
                                             facecolor=facecolor,
                                             edgecolor=edgecolor,
                                             frameon=frameon,
                                             FigureClass=FigureClass,
                                             **kwargs)

        # make this figure current on button press event
        def make_active(event):
            _pylab_helpers.Gcf.set_active(figManager)

        cid = figManager.canvas.mpl_connect('button_press_event', make_active)
        figManager._cidgcf = cid

        _pylab_helpers.Gcf.set_active(figManager)
        figManager.canvas.figure.number = num

    draw_if_interactive()
    return figManager.canvas.figure

def gcf():
    "Return a reference to the current figure."

    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is not None:
        return figManager.canvas.figure
    else:
        return figure()

fignum_exists = _pylab_helpers.Gcf.has_fignum

def get_fignums():
    "Return a list of existing figure numbers."
    fignums = _pylab_helpers.Gcf.figs.keys()
    fignums.sort()
    return fignums

def get_current_fig_manager():
    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is None:
        gcf()  # creates an active figure as a side effect
        figManager = _pylab_helpers.Gcf.get_active()
    return figManager

# note we check for __doc__ is not None since py2exe optimize removes
# the docstrings
def connect(s, func):
    return get_current_fig_manager().canvas.mpl_connect(s, func)
if FigureCanvasBase.mpl_connect.__doc__ is not None:
    connect.__doc__ = dedent(FigureCanvasBase.mpl_connect.__doc__)

def disconnect(cid):
    return get_current_fig_manager().canvas.mpl_disconnect(cid)
if FigureCanvasBase.mpl_disconnect.__doc__ is not None:
    disconnect.__doc__ = dedent(FigureCanvasBase.mpl_disconnect.__doc__)

def close(*args):
    """
    Close a figure window

    ``close()`` by itself closes the current figure

    ``close(num)`` closes figure number *num*

    ``close(h)`` where *h* is a :class:`Figure` instance, closes that figure

    ``close('all')`` closes all the figure windows
    """

    if len(args)==0:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is None: return
        else:
            figManager.canvas.mpl_disconnect(figManager._cidgcf)
            _pylab_helpers.Gcf.destroy(figManager.num)
    elif len(args)==1:
        arg = args[0]
        if arg=='all':
            for manager in _pylab_helpers.Gcf.get_all_fig_managers():
                manager.canvas.mpl_disconnect(manager._cidgcf)
                _pylab_helpers.Gcf.destroy(manager.num)
        elif isinstance(arg, int):
            _pylab_helpers.Gcf.destroy(arg)
        elif isinstance(arg, Figure):
            for manager in _pylab_helpers.Gcf.get_all_fig_managers():
                if manager.canvas.figure==arg:
                    manager.canvas.mpl_disconnect(manager._cidgcf)
                    _pylab_helpers.Gcf.destroy(manager.num)
        else:
            raise TypeError('Unrecognized argument type %s to close'%type(arg))
    else:
        raise TypeError('close takes 0 or 1 arguments')


def clf():
    """
    Clear the current figure
    """
    gcf().clf()
    draw_if_interactive()

def draw():
    'redraw the current figure'
    get_current_fig_manager().canvas.draw()

def savefig(*args, **kwargs):
    fig = gcf()
    return fig.savefig(*args, **kwargs)
if Figure.savefig.__doc__ is not None:
    savefig.__doc__ = dedent(Figure.savefig.__doc__)

def ginput(*args, **kwargs):
    """
    Blocking call to interact with the figure.

    This will wait for *n* clicks from the user and return a list of the
    coordinates of each click.

    If *timeout* is negative, does not timeout.
    """
    return gcf().ginput(*args, **kwargs)
if Figure.ginput.__doc__ is not None:
    ginput.__doc__ = dedent(Figure.ginput.__doc__)

def waitforbuttonpress(*args, **kwargs):
    """
    Blocking call to interact with the figure.

    This will wait for *n* key or mouse clicks from the user and
    return a list containing True's for keyboard clicks and False's
    for mouse clicks.

    If *timeout* is negative, does not timeout.
    """
    return gcf().waitforbuttonpress(*args, **kwargs)
if Figure.waitforbuttonpress.__doc__ is not None:
    waitforbuttonpress.__doc__ = dedent(Figure.waitforbuttonpress.__doc__)


# Putting things in figures

def figtext(*args, **kwargs):

    ret =  gcf().text(*args, **kwargs)
    draw_if_interactive()
    return ret
if Figure.text.__doc__ is not None:
    figtext.__doc__ = dedent(Figure.text.__doc__)

def suptitle(*args, **kwargs):
    ret =  gcf().suptitle(*args, **kwargs)
    draw_if_interactive()
    return ret
if Figure.suptitle.__doc__ is not None:
    suptitle.__doc__ = dedent(Figure.suptitle.__doc__)

def figimage(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    ret =  gcf().figimage(*args, **kwargs)
    draw_if_interactive()
    gci._current = ret
    return ret
if Figure.figimage.__doc__ is not None:
    figimage.__doc__ = dedent(Figure.figimage.__doc__) + """
Addition kwargs: hold = [True|False] overrides default hold state"""

def figlegend(handles, labels, loc, **kwargs):
    """
    Place a legend in the figure.

    *labels*
      a sequence of strings

    *handles*
      a sequence of :class:`~matplotlib.lines.Line2D` or
      :class:`~matplotlib.patches.Patch` instances

    *loc*
      can be a string or an integer specifying the legend
      location

    A :class:`matplotlib.legend.Legend` instance is returned.

    Example::

      figlegend( (line1, line2, line3),
                 ('label1', 'label2', 'label3'),
                 'upper right' )

    .. seealso::

       :func:`~matplotlib.pyplot.legend`

    """
    l = gcf().legend(handles, labels, loc, **kwargs)
    draw_if_interactive()
    return l



## Figure and Axes hybrid ##

def hold(b=None):
    """
    Set the hold state.  If *b* is None (default), toggle the
    hold state, else set the hold state to boolean value *b*::

      hold()      # toggle hold
      hold(True)  # hold is on
      hold(False) # hold is off

    When *hold* is *True*, subsequent plot commands will be added to
    the current axes.  When *hold* is *False*, the current axes and
    figure will be cleared on the next plot command.
    """

    fig = gcf()
    ax = fig.gca()

    fig.hold(b)
    ax.hold(b)

    # b=None toggles the hold state, so let's get get the current hold
    # state; but should pyplot hold toggle the rc setting - me thinks
    # not
    b = ax.ishold()

    rc('axes', hold=b)

def ishold():
    """
    Return the hold status of the current axes
    """
    return gca().ishold()

def over(func, *args, **kwargs):
    """
    over calls::

      func(*args, **kwargs)

    with ``hold(True)`` and then restores the hold state.
    """
    h = ishold()
    hold(True)
    func(*args, **kwargs)
    hold(h)



## Axes ##

def axes(*args, **kwargs):
    """
    Add an axes at position rect specified by:

    - ``axes()`` by itself creates a default full ``subplot(111)`` window axis.

    - ``axes(rect, axisbg='w')`` where *rect* = [left, bottom, width,
      height] in normalized (0, 1) units.  *axisbg* is the background
      color for the axis, default white.

    - ``axes(h)`` where *h* is an axes instance makes *h* the current
      axis.  An :class:`~matplotlib.axes.Axes` instance is returned.

    =======   ============   ================================================
    kwarg     Accepts        Desctiption
    =======   ============   ================================================
    axisbg    color          the axes background color
    frameon   [True|False]   display the frame?
    sharex    otherax        current axes shares xaxis attribute with otherax
    sharey    otherax        current axes shares yaxis attribute with otherax
    polar     [True|False]   use a polar axes?
    =======   ============   ================================================

    Examples:

    * :file:`examples/pylab_examples/axes_demo.py` places custom axes.
    * :file:`examples/pylab_examples/shared_axis_demo.py` uses
      *sharex* and *sharey*.

    """

    nargs = len(args)
    if len(args)==0: return subplot(111, **kwargs)
    if nargs>1:
        raise TypeError('Only one non keyword arg to axes allowed')
    arg = args[0]

    if isinstance(arg, Axes):
        a = gcf().sca(arg)
    else:
        rect = arg
        a = gcf().add_axes(rect, **kwargs)
    draw_if_interactive()
    return a

def delaxes(*args):
    """
    ``delaxes(ax)``: remove *ax* from the current figure.  If *ax*
    doesn't exist, an error will be raised.

    ``delaxes()``: delete the current axes
    """
    if not len(args):
        ax = gca()
    else:
        ax = args[0]
    ret = gcf().delaxes(ax)
    draw_if_interactive()
    return ret




def gca(**kwargs):
    """
    Return the current axis instance.  This can be used to control
    axis properties either using set or the
    :class:`~matplotlib.axes.Axes` methods, for example, setting the
    xaxis range::

      plot(t,s)
      set(gca(), 'xlim', [0,10])

    or::

      plot(t,s)
      a = gca()
      a.set_xlim([0,10])

    """

    ax =  gcf().gca(**kwargs)
    return ax

# More ways of creating axes:

def subplot(*args, **kwargs):
    """
    Create a subplot command, creating axes with::

      subplot(numRows, numCols, plotNum)

    where *plotNum* = 1 is the first plot number and increasing *plotNums*
    fill rows first.  max(*plotNum*) == *numRows* * *numCols*

    You can leave out the commas if *numRows* <= *numCols* <=
    *plotNum* < 10, as in::

      subplot(211)    # 2 rows, 1 column, first (upper) plot

    ``subplot(111)`` is the default axis.

    New subplots that overlap old will delete the old axes.  If you do
    not want this behavior, use
    :meth:`matplotlib.figure.Figure.add_subplot` or the
    :func:`~matplotlib.pyplot.axes` command.  Eg.::

      from pylab import *
      plot([1,2,3])  # implicitly creates subplot(111)
      subplot(211)   # overlaps, subplot(111) is killed
      plot(rand(12), rand(12))
      subplot(212, axisbg='y') # creates 2nd subplot with yellow background

    Keyword arguments:

      *axisbg*:
        The background color of the subplot, which can be any valid
        color specifier.  See :mod:`matplotlib.colors` for more
        information.

      *polar*:
        A boolean flag indicating whether the subplot plot should be
        a polar projection.  Defaults to False.

      *projection*:
        A string giving the name of a custom projection to be used
        for the subplot. This projection must have been previously
        registered. See :func:`matplotlib.projections.register_projection`

    .. seealso::

        :func:`~matplotlib.pyplot.axes`
            For additional information on :func:`axes` and
            :func:`subplot` keyword arguments.

        :file:`examples/pylab_examples/polar_scatter.py`
            For an example

    **Example:**

    .. plot:: mpl_examples/pylab_examples/subplot_demo.py

    """


    fig = gcf()
    a = fig.add_subplot(*args, **kwargs)
    bbox = a.bbox
    byebye = []
    for other in fig.axes:
        if other==a: continue
        if bbox.fully_overlaps(other.bbox):
            byebye.append(other)
    for ax in byebye: delaxes(ax)

    draw_if_interactive()
    return a


def twinx(ax=None):
    """
    Make a second axes overlay *ax* (or the current axes if *ax* is
    *None*) sharing the xaxis.  The ticks for *ax2* will be placed on
    the right, and the *ax2* instance is returned.

    .. seealso::

       :file:`examples/api_examples/two_scales.py`
          For an example
    """
    if ax is None:
        ax=gca()
    ax1 = ax.twinx()
    draw_if_interactive()
    return ax1


def twiny(ax=None):
    """
    Make a second axes overlay *ax* (or the current axes if *ax* is
    *None*) sharing the yaxis.  The ticks for *ax2* will be placed on
    the top, and the *ax2* instance is returned.
    """
    if ax is None:
        ax=gca()
    ax1 = ax.twiny()
    draw_if_interactive()
    return ax1



def subplots_adjust(*args, **kwargs):
    """
    call signature::

      subplots_adjust(left=None, bottom=None, right=None, top=None,
                      wspace=None, hspace=None)

    Tune the subplot layout via the
    :class:`matplotlib.figure.SubplotParams` mechanism.  The parameter
    meanings (and suggested defaults) are::

      left  = 0.125  # the left side of the subplots of the figure
      right = 0.9    # the right side of the subplots of the figure
      bottom = 0.1   # the bottom of the subplots of the figure
      top = 0.9      # the top of the subplots of the figure
      wspace = 0.2   # the amount of width reserved for blank space between subplots
      hspace = 0.2   # the amount of height reserved for white space between subplots

    The actual defaults are controlled by the rc file
    """
    fig = gcf()
    fig.subplots_adjust(*args, **kwargs)
    draw_if_interactive()


def subplot_tool(targetfig=None):
    """
    Launch a subplot tool window for *targetfig* (default gcf).

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
    Turn the axes box on or off according to *on*.

    If *on* is *None*, toggle state.
    """
    ax = gca()
    if on is None:
        on = not ax.get_frame_on()
    ax.set_frame_on(on)
    draw_if_interactive()

def title(s, *args, **kwargs):
    """
    Set the title of the current axis to *s*.

    Default font override is::

      override = {'fontsize': 'medium',
                  'verticalalignment': 'bottom',
                  'horizontalalignment': 'center'}

    .. seealso::

       :func:`~matplotlib.pyplot.text`
           for information on how override and the optional args work.
    """
    l =  gca().set_title(s, *args, **kwargs)
    draw_if_interactive()
    return l




## Axis ##

def axis(*v, **kwargs):
    """
    Set/Get the axis properties:

      >>> axis()

    returns the current axes limits ``[xmin, xmax, ymin, ymax]``.

      >>> axis(v)

    sets the min and max of the x and y axes, with
    ``v = [xmin, xmax, ymin, ymax]``.

      >>> axis('off')

    turns off the axis lines and labels.

      >>> axis('equal')

    changes limits of *x* or *y* axis so that equal increments of *x*
    and *y* have the same length; a circle is circular.

      >>> axis('scaled')

    achieves the same result by changing the dimensions of the plot box instead
    of the axis data limits.

      >>> axis('tight')

    changes *x* and *y* axis limits such that all data is shown. If
    all data is already shown, it will move it to the center of the
    figure without modifying (*xmax* - *xmin*) or (*ymax* -
    *ymin*). Note this is slightly different than in matlab.

      >>> axis('image')

    is 'scaled' with the axis limits equal to the data limits.

      >>> axis('auto')

    and

      >>> axis('normal')

    are deprecated. They restore default behavior; axis limits are automatically
    scaled to make the data fit comfortably within the plot box.

    if ``len(*v)==0``, you can pass in *xmin*, *xmax*, *ymin*, *ymax*
    as kwargs selectively to alter just those limits without changing
    the others.

    The xmin, xmax, ymin, ymax tuple is returned

    .. seealso::

        :func:`xlim`, :func:`ylim`
           For setting the x- and y-limits individually.
    """
    ax = gca()
    v = ax.axis(*v, **kwargs)
    draw_if_interactive()
    return v

def xlabel(s, *args, **kwargs):
    """
    Set the *x* axis label of the current axis to *s*

    Default override is::

      override = {
          'fontsize'            : 'small',
          'verticalalignment'   : 'top',
          'horizontalalignment' : 'center'
          }

    .. seealso::

        :func:`~matplotlib.pyplot.text`
            For information on how override and the optional args work
    """
    l =  gca().set_xlabel(s, *args, **kwargs)
    draw_if_interactive()
    return l

def ylabel(s, *args, **kwargs):
    """
    Set the *y* axis label of the current axis to *s*.

    Defaults override is::

        override = {
           'fontsize'            : 'small',
           'verticalalignment'   : 'center',
           'horizontalalignment' : 'right',
           'rotation'='vertical' : }

    .. seealso::

        :func:`~matplotlib.pyplot.text`
            For information on how override and the optional args
            work.
    """
    l = gca().set_ylabel(s, *args, **kwargs)
    draw_if_interactive()
    return l





def xlim(*args, **kwargs):
    """
    Set/Get the xlimits of the current axes::

      xmin, xmax = xlim()   # return the current xlim
      xlim( (xmin, xmax) )  # set the xlim to xmin, xmax
      xlim( xmin, xmax )    # set the xlim to xmin, xmax

    If you do not specify args, you can pass the xmin and xmax as
    kwargs, eg.::

      xlim(xmax=3) # adjust the max leaving min unchanged
      xlim(xmin=1) # adjust the min leaving max unchanged

    The new axis limits are returned as a length 2 tuple.

    """
    ax = gca()
    ret = ax.set_xlim(*args, **kwargs)
    draw_if_interactive()
    return ret


def ylim(*args, **kwargs):
    """
    Set/Get the ylimits of the current axes::

      ymin, ymax = ylim()   # return the current ylim
      ylim( (ymin, ymax) )  # set the ylim to ymin, ymax
      ylim( ymin, ymax )    # set the ylim to ymin, ymax

    If you do not specify args, you can pass the *ymin* and *ymax* as
    kwargs, eg.::

      ylim(ymax=3) # adjust the max leaving min unchanged
      ylim(ymin=1) # adjust the min leaving max unchanged

    The new axis limits are returned as a length 2 tuple.
    """
    ax = gca()
    ret = ax.set_ylim(*args, **kwargs)
    draw_if_interactive()
    return ret


def xscale(*args, **kwargs):
    """
    call signature::

      xscale(scale, **kwargs)

    Set the scaling for the x-axis: %(scale)s

    Different keywords may be accepted, depending on the scale:

    %(scale_docs)s
    """
    ax = gca()
    ret = ax.set_xscale(*args, **kwargs)
    draw_if_interactive()
    return ret
xscale.__doc__ = dedent(xscale.__doc__) % {
    'scale': ' | '.join([repr(_x) for _x in get_scale_names()]),
    'scale_docs': get_scale_docs()}

def yscale(*args, **kwargs):
    """
    call signature::

      xscale(scale, **kwargs)

    Set the scaling for the y-axis: %(scale)s

    Different keywords may be accepted, depending on the scale:

    %(scale_docs)s
    """
    ax = gca()
    ret = ax.set_yscale(*args, **kwargs)
    draw_if_interactive()
    return ret
yscale.__doc__ = dedent(yscale.__doc__) % {
    'scale': ' | '.join([repr(_x) for _x in get_scale_names()]),
    'scale_docs': get_scale_docs()}

def xticks(*args, **kwargs):
    """
    Set/Get the xlimits of the current ticklocs and labels::

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

    if len(args)==0:
        locs = ax.get_xticks()
        labels = ax.get_xticklabels()
    elif len(args)==1:
        locs = ax.set_xticks(args[0])
        labels = ax.get_xticklabels()
    elif len(args)==2:
        locs = ax.set_xticks(args[0])
        labels = ax.set_xticklabels(args[1], **kwargs)
    else: raise TypeError('Illegal number of arguments to xticks')
    if len(kwargs):
        for l in labels:
            l.update(kwargs)

    draw_if_interactive()
    return locs, silent_list('Text xticklabel', labels)

def yticks(*args, **kwargs):
    """
    Set/Get the ylimits of the current ticklocs and labels::

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

    if len(args)==0:
        locs = ax.get_yticks()
        labels = ax.get_yticklabels()
    elif len(args)==1:
        locs = ax.set_yticks(args[0])
        labels = ax.get_yticklabels()
    elif len(args)==2:
        locs = ax.set_yticks(args[0])
        labels = ax.set_yticklabels(args[1], **kwargs)
    else: raise TypeError('Illegal number of arguments to yticks')
    if len(kwargs):
        for l in labels:
            l.update(kwargs)

    draw_if_interactive()

    return ( locs,
             silent_list('Text yticklabel', labels)
             )

def minorticks_on():
    """
    Display minor ticks on the current plot.

    Displaying minor ticks reduces performance; turn them off using
    minorticks_off() if drawing speed is a problem.
    """
    gca().minorticks_on()
    draw_if_interactive()

def minorticks_off():
    """
    Remove minor ticks from the current plot.
    """
    gca().minorticks_off()
    draw_if_interactive()

def rgrids(*args, **kwargs):
    """
    Set/Get the radial locations of the gridlines and ticklabels on a
    polar plot.

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
        lines = ax.yaxis.get_ticklines()
        labels = ax.yaxis.get_ticklabels()
    else:
        lines, labels = ax.set_rgrids(*args, **kwargs)

    draw_if_interactive()
    return ( silent_list('Line2D rgridline', lines),
             silent_list('Text rgridlabel', labels) )

def thetagrids(*args, **kwargs):
    """
    Set/Get the theta locations of the gridlines and ticklabels.

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
    the label (1 is the edge). Eg. 1.05 is outside the axes and 0.95
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

    draw_if_interactive()
    return (silent_list('Line2D thetagridline', lines),
            silent_list('Text thetagridlabel', labels)
            )


## Plotting Info ##

def plotting():
    """
    Plotting commands

    =============== =========================================================
    Command         Description
    =============== =========================================================
    axes            Create a new axes
    axis            Set or return the current axis limits
    bar             make a bar chart
    boxplot         make a box and whiskers chart
    cla             clear current axes
    clabel          label a contour plot
    clf             clear a figure window
    close           close a figure window
    colorbar        add a colorbar to the current figure
    cohere          make a plot of coherence
    contour         make a contour plot
    contourf        make a filled contour plot
    csd             make a plot of cross spectral density
    draw            force a redraw of the current figure
    errorbar        make an errorbar graph
    figlegend       add a legend to the figure
    figimage        add an image to the figure, w/o resampling
    figtext         add text in figure coords
    figure          create or change active figure
    fill            make filled polygons
    fill_between    make filled polygons between two sets of y-values
    fill_betweenx   make filled polygons between two sets of x-values
    gca             return the current axes
    gcf             return the current figure
    gci             get the current image, or None
    getp            get a graphics property
    hist            make a histogram
    hold            set the hold state on current axes
    legend          add a legend to the axes
    loglog          a log log plot
    imread          load image file into array
    imsave          save array as an image file
    imshow          plot image data
    matshow         display a matrix in a new figure preserving aspect
    pcolor          make a pseudocolor plot
    plot            make a line plot
    plotfile        plot data from a flat file
    psd             make a plot of power spectral density
    quiver          make a direction field (arrows) plot
    rc              control the default params
    savefig         save the current figure
    scatter         make a scatter plot
    setp            set a graphics property
    semilogx        log x axis
    semilogy        log y axis
    show            show the figures
    specgram        a spectrogram plot
    stem            make a stem plot
    subplot         make a subplot (numrows, numcols, axesnum)
    table           add a table to the axes
    text            add some text at location x,y to the current axes
    title           add a title to the current axes
    xlabel          add an xlabel to the current axes
    ylabel          add a ylabel to the current axes
    =============== =========================================================

    The following commands will set the default colormap accordingly:

    * autumn
    * bone
    * cool
    * copper
    * flag
    * gray
    * hot
    * hsv
    * jet
    * pink
    * prism
    * spring
    * summer
    * winter
    * spectral

    """
    pass


def get_plot_commands(): return ( 'axes', 'axis', 'bar', 'boxplot', 'cla', 'clf',
    'close', 'colorbar', 'cohere', 'csd', 'draw', 'errorbar',
    'figlegend', 'figtext', 'figimage', 'figure', 'fill', 'gca',
    'gcf', 'gci', 'get', 'gray', 'barh', 'jet', 'hist', 'hold', 'imread', 'imsave',
    'imshow', 'legend', 'loglog', 'quiver', 'rc', 'pcolor', 'pcolormesh', 'plot', 'psd',
    'savefig', 'scatter', 'set', 'semilogx', 'semilogy', 'show',
    'specgram', 'stem', 'subplot', 'table', 'text', 'title', 'xlabel',
    'ylabel', 'pie', 'polar')

def colors():
    """
    This is a do-nothing function to provide you with help on how
    matplotlib handles colors.

    Commands which take color arguments can use several formats to
    specify the colors.  For the basic builtin colors, you can use a
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

      color = 'red',
      color = 'burlywood'
      color = 'chartreuse'

    The example below creates a subplot with a dark
    slate gray background

       subplot(111, axisbg=(0.1843, 0.3098, 0.3098))

    Here is an example that creates a pale turqoise title::

      title('Is this the best color?', color='#afeeee')

    """
    pass



def colormaps():
    """
    matplotlib provides the following colormaps.

    * autumn
    * bone
    * cool
    * copper
    * flag
    * gray
    * hot
    * hsv
    * jet
    * pink
    * prism
    * spring
    * summer
    * winter
    * spectral

    You can set the colormap for an image, pcolor, scatter, etc,
    either as a keyword argument::

      imshow(X, cmap=cm.hot)

    or post-hoc using the corresponding pylab interface function::

      imshow(X)
      hot()
      jet()

    In interactive mode, this will update the colormap allowing you to
    see which one works best for your data.
    """
    pass


## Plotting part 1: manually generated functions and wrappers ##


from matplotlib.colorbar import colorbar_doc
def colorbar(mappable=None, cax=None, ax=None, **kw):
    if mappable is None:
        mappable = gci()
    if ax is None:
        ax = gca()

    ret = gcf().colorbar(mappable, cax = cax, ax=ax, **kw)
    draw_if_interactive()
    return ret
colorbar.__doc__ = colorbar_doc

def clim(vmin=None, vmax=None):
    """
    Set the color limits of the current image

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
        raise RuntimeError('You must first define an image, eg with imshow')

    im.set_clim(vmin, vmax)
    draw_if_interactive()



def imread(*args, **kwargs):
    return _imread(*args, **kwargs)
if _imread.__doc__ is not None:
    imread.__doc__ = dedent(_imread.__doc__)

def imsave(*args, **kwargs):
    return _imsave(*args, **kwargs)
if _imsave.__doc__ is not None:
    imsave.__doc__ = dedent(_imsave.__doc__)

def matshow(A, fignum=None, **kw):
    """
    Display an array as a matrix in a new figure window.

    The origin is set at the upper left hand corner and rows (first
    dimension of the array) are displayed horizontally.  The aspect
    ratio of the figure window is that of the array, unless this would
    make an excessively short or narrow figure.

    Tick labels for the xaxis are placed on top.

    With the exception of fignum, keyword arguments are passed to
    :func:`~matplotlib.pyplot.imshow`.


    *fignum*: [ None | integer | False ]
      By default, :func:`matshow` creates a new figure window with
      automatic numbering.  If *fignum* is given as an integer, the
      created figure will use this figure number.  Because of how
      :func:`matshow` tries to set the figure aspect ratio to be the
      one of the array, if you provide the number of an already
      existing figure, strange things may happen.

      If *fignum* is *False* or 0, a new figure window will **NOT** be created.
    """
    if fignum is False or fignum is 0:
        ax = gca()
    else:
        # Extract actual aspect ratio of array and make appropriately sized figure
        fig = figure(fignum, figsize=figaspect(A))
        ax  = fig.add_axes([0.15, 0.09, 0.775, 0.775])

    im = ax.matshow(A, **kw)
    gci._current = im

    draw_if_interactive()
    return im

def polar(*args, **kwargs):
    """
    call signature::

      polar(theta, r, **kwargs)

    Make a polar plot.  Multiple *theta*, *r* arguments are supported,
    with format strings, as in :func:`~matplotlib.pyplot.plot`.

    An optional kwarg *resolution* sets the number of vertices to
    interpolate between each pair of points.  The default is 1,
    which disables interpolation.
    """
    resolution = kwargs.pop('resolution', None)
    ax = gca(polar=True, resolution=resolution)
    ret = ax.plot(*args, **kwargs)
    draw_if_interactive()
    return ret

def plotfile(fname, cols=(0,), plotfuncs=None,
             comments='#', skiprows=0, checkrows=5, delimiter=',', names=None,
             subplots=True, newfig=True,
             **kwargs):
    """
    Plot the data in *fname*

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
    vector as you use in the *plotfuncs* dictionary, eg., integer
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

    if len(cols)<1:
        raise ValueError('must have at least one column of data')

    if plotfuncs is None:
        plotfuncs = dict()
    r = mlab.csv2rec(fname, comments=comments, skiprows=skiprows,
                     checkrows=checkrows, delimiter=delimiter, names=names)

    def getname_val(identifier):
        'return the name and column data for identifier'
        if is_string_like(identifier):
            return identifier, r[identifier]
        elif is_numlike(identifier):
            name = r.dtype.names[int(identifier)]
            return name, r[name]
        else:
            raise TypeError('identifier must be a string or integer')

    xname, x = getname_val(cols[0])
    ynamelist = []

    if len(cols)==1:
        ax1 = fig.add_subplot(1,1,1)
        funcname = plotfuncs.get(cols[0], 'plot')
        func = getattr(ax1, funcname)
        func(x, **kwargs)
        ax1.set_ylabel(xname)
    else:
        N = len(cols)
        for i in range(1,N):
            if subplots:
                if i==1:
                    ax = ax1 = fig.add_subplot(N-1,1,i)
                else:
                    ax = fig.add_subplot(N-1,1,i, sharex=ax1)
            elif i==1:
                ax = fig.add_subplot(1,1,1)

            ax.grid(True)


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

    if xname=='date':
        fig.autofmt_xdate()

    draw_if_interactive()


## Plotting part 2: autogenerated wrappers for axes methods ##


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def acorr(x, hold=None, **kwargs):
    """call signature::

    acorr(x, normed=True, detrend=mlab.detrend_none, usevlines=True,
          maxlags=10, **kwargs)

Plot the autocorrelation of *x*.  If *normed* = *True*,
normalize the data by the autocorrelation at 0-th lag.  *x* is
detrended by the *detrend* callable (default no normalization).

Data are plotted as ``plot(lags, c, **kwargs)``

Return value is a tuple (*lags*, *c*, *line*) where:

  - *lags* are a length 2*maxlags+1 lag vector

  - *c* is the 2*maxlags+1 auto correlation vector

  - *line* is a :class:`~matplotlib.lines.Line2D` instance
    returned by :meth:`plot`

The default *linestyle* is None and the default *marker* is
``'o'``, though these can be overridden with keyword args.
The cross correlation is performed with
:func:`numpy.correlate` with *mode* = 2.

If *usevlines* is *True*, :meth:`~matplotlib.axes.Axes.vlines`
rather than :meth:`~matplotlib.axes.Axes.plot` is used to draw
vertical lines from the origin to the acorr.  Otherwise, the
plot style is determined by the kwargs, which are
:class:`~matplotlib.lines.Line2D` properties.

*maxlags* is a positive integer detailing the number of lags
to show.  The default value of *None* will return all
:math:`2 \\mathrm{len}(x) - 1` lags.

The return value is a tuple (*lags*, *c*, *linecol*, *b*)
where

- *linecol* is the
  :class:`~matplotlib.collections.LineCollection`

- *b* is the *x*-axis.

.. seealso::

    :meth:`~matplotlib.axes.Axes.plot` or
    :meth:`~matplotlib.axes.Axes.vlines`
       For documentation on valid kwargs.

**Example:**

:func:`~matplotlib.pyplot.xcorr` above, and
:func:`~matplotlib.pyplot.acorr` below.

**Example:**

.. plot:: mpl_examples/pylab_examples/xcorr_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.acorr(x, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def arrow(x, y, dx, dy, hold=None, **kwargs):
    """call signature::

   arrow(x, y, dx, dy, **kwargs)

Draws arrow on specified axis from (*x*, *y*) to (*x* + *dx*,
*y* + *dy*).

Optional kwargs control the arrow properties:
  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/arrow_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.arrow(x, y, dx, dy, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axhline(y=0, xmin=0, xmax=1, hold=None, **kwargs):
    """call signature::

  axhline(y=0, xmin=0, xmax=1, **kwargs)

Axis Horizontal Line

Draw a horizontal line at *y* from *xmin* to *xmax*.  With the
default values of *xmin* = 0 and *xmax* = 1, this line will
always span the horizontal extent of the axes, regardless of
the xlim settings, even if you change them, eg. with the
:meth:`set_xlim` command.  That is, the horizontal extent is
in axes coords: 0=left, 0.5=middle, 1.0=right but the *y*
location is in data coordinates.

Return value is the :class:`~matplotlib.lines.Line2D`
instance.  kwargs are the same as kwargs to plot, and can be
used to control the line properties.  Eg.,

* draw a thick red hline at *y* = 0 that spans the xrange

    >>> axhline(linewidth=4, color='r')

* draw a default hline at *y* = 1 that spans the xrange

    >>> axhline(y=1)

* draw a default hline at *y* = .5 that spans the the middle half of
  the xrange

    >>> axhline(y=.5, xmin=0.25, xmax=0.75)

Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

.. seealso::

    :meth:`axhspan`
        for example plot and source code

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.axhline(y, xmin, xmax, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axhspan(ymin, ymax, xmin=0, xmax=1, hold=None, **kwargs):
    """call signature::

  axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)

Axis Horizontal Span.

*y* coords are in data units and *x* coords are in axes (relative
0-1) units.

Draw a horizontal span (rectangle) from *ymin* to *ymax*.
With the default values of *xmin* = 0 and *xmax* = 1, this
always spans the xrange, regardless of the xlim settings, even
if you change them, eg. with the :meth:`set_xlim` command.
That is, the horizontal extent is in axes coords: 0=left,
0.5=middle, 1.0=right but the *y* location is in data
coordinates.

Return value is a :class:`matplotlib.patches.Polygon`
instance.

Examples:

* draw a gray rectangle from *y* = 0.25-0.75 that spans the
  horizontal extent of the axes

    >>> axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

Valid kwargs are :class:`~matplotlib.patches.Polygon` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/axhspan_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.axhspan(ymin, ymax, xmin, xmax, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axvline(x=0, ymin=0, ymax=1, hold=None, **kwargs):
    """call signature::

  axvline(x=0, ymin=0, ymax=1, **kwargs)

Axis Vertical Line

Draw a vertical line at *x* from *ymin* to *ymax*.  With the
default values of *ymin* = 0 and *ymax* = 1, this line will
always span the vertical extent of the axes, regardless of the
ylim settings, even if you change them, eg. with the
:meth:`set_ylim` command.  That is, the vertical extent is in
axes coords: 0=bottom, 0.5=middle, 1.0=top but the *x* location
is in data coordinates.

Return value is the :class:`~matplotlib.lines.Line2D`
instance.  kwargs are the same as kwargs to plot, and can be
used to control the line properties.  Eg.,

* draw a thick red vline at *x* = 0 that spans the yrange

    >>> axvline(linewidth=4, color='r')

* draw a default vline at *x* = 1 that spans the yrange

    >>> axvline(x=1)

* draw a default vline at *x* = .5 that spans the the middle half of
  the yrange

    >>> axvline(x=.5, ymin=0.25, ymax=0.75)

Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

.. seealso::

    :meth:`axhspan`
        for example plot and source code

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.axvline(x, ymin, ymax, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axvspan(xmin, xmax, ymin=0, ymax=1, hold=None, **kwargs):
    """call signature::

  axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)

Axis Vertical Span.

*x* coords are in data units and *y* coords are in axes (relative
0-1) units.

Draw a vertical span (rectangle) from *xmin* to *xmax*.  With
the default values of *ymin* = 0 and *ymax* = 1, this always
spans the yrange, regardless of the ylim settings, even if you
change them, eg. with the :meth:`set_ylim` command.  That is,
the vertical extent is in axes coords: 0=bottom, 0.5=middle,
1.0=top but the *y* location is in data coordinates.

Return value is the :class:`matplotlib.patches.Polygon`
instance.

Examples:

* draw a vertical green translucent rectangle from x=1.25 to 1.55 that
  spans the yrange of the axes

    >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

Valid kwargs are :class:`~matplotlib.patches.Polygon`
properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

.. seealso::

    :meth:`axhspan`
        for example plot and source code

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.axvspan(xmin, xmax, ymin, ymax, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def bar(left, height, width=0.80000000000000004, bottom=None, color=None, edgecolor=None, linewidth=None, yerr=None, xerr=None, ecolor=None, capsize=3, align='edge', orientation='vertical', log=False, hold=None, **kwargs):
    """call signature::

  bar(left, height, width=0.8, bottom=0,
      color=None, edgecolor=None, linewidth=None,
      yerr=None, xerr=None, ecolor=None, capsize=3,
      align='edge', orientation='vertical', log=False)

Make a bar plot with rectangles bounded by:

  *left*, *left* + *width*, *bottom*, *bottom* + *height*
        (left, right, bottom and top edges)

*left*, *height*, *width*, and *bottom* can be either scalars
or sequences

Return value is a list of
:class:`matplotlib.patches.Rectangle` instances.

Required arguments:

  ========   ===============================================
  Argument   Description
  ========   ===============================================
  *left*     the x coordinates of the left sides of the bars
  *height*   the heights of the bars
  ========   ===============================================

Optional keyword arguments:

  ===============   ==========================================
  Keyword           Description
  ===============   ==========================================
  *width*           the widths of the bars
  *bottom*          the y coordinates of the bottom edges of
                    the bars
  *color*           the colors of the bars
  *edgecolor*       the colors of the bar edges
  *linewidth*       width of bar edges; None means use default
                    linewidth; 0 means don't draw edges.
  *xerr*            if not None, will be used to generate
                    errorbars on the bar chart
  *yerr*            if not None, will be used to generate
                    errorbars on the bar chart
  *ecolor*          specifies the color of any errorbar
  *capsize*         (default 3) determines the length in
                    points of the error bar caps
  *align*           'edge' (default) | 'center'
  *orientation*     'vertical' | 'horizontal'
  *log*             [False|True] False (default) leaves the
                    orientation axis as-is; True sets it to
                    log scale
  ===============   ==========================================

For vertical bars, *align* = 'edge' aligns bars by their left
edges in left, while *align* = 'center' interprets these
values as the *x* coordinates of the bar centers. For
horizontal bars, *align* = 'edge' aligns bars by their bottom
edges in bottom, while *align* = 'center' interprets these
values as the *y* coordinates of the bar centers.

The optional arguments *color*, *edgecolor*, *linewidth*,
*xerr*, and *yerr* can be either scalars or sequences of
length equal to the number of bars.  This enables you to use
bar as the basis for stacked bar charts, or candlestick plots.

Other optional kwargs:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

**Example:** A stacked bar chart.

.. plot:: mpl_examples/pylab_examples/bar_stacked.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.bar(left, height, width, bottom, color, edgecolor, linewidth, yerr, xerr, ecolor, capsize, align, orientation, log, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def barh(bottom, width, height=0.80000000000000004, left=None, hold=None, **kwargs):
    """call signature::

  barh(bottom, width, height=0.8, left=0, **kwargs)

Make a horizontal bar plot with rectangles bounded by:

  *left*, *left* + *width*, *bottom*, *bottom* + *height*
        (left, right, bottom and top edges)

*bottom*, *width*, *height*, and *left* can be either scalars
or sequences

Return value is a list of
:class:`matplotlib.patches.Rectangle` instances.

Required arguments:

  ========   ======================================================
  Argument   Description
  ========   ======================================================
  *bottom*   the vertical positions of the bottom edges of the bars
  *width*    the lengths of the bars
  ========   ======================================================

Optional keyword arguments:

  ===============   ==========================================
  Keyword           Description
  ===============   ==========================================
  *height*          the heights (thicknesses) of the bars
  *left*            the x coordinates of the left edges of the
                    bars
  *color*           the colors of the bars
  *edgecolor*       the colors of the bar edges
  *linewidth*       width of bar edges; None means use default
                    linewidth; 0 means don't draw edges.
  *xerr*            if not None, will be used to generate
                    errorbars on the bar chart
  *yerr*            if not None, will be used to generate
                    errorbars on the bar chart
  *ecolor*          specifies the color of any errorbar
  *capsize*         (default 3) determines the length in
                    points of the error bar caps
  *align*           'edge' (default) | 'center'
  *log*             [False|True] False (default) leaves the
                    horizontal axis as-is; True sets it to log
                    scale
  ===============   ==========================================

Setting *align* = 'edge' aligns bars by their bottom edges in
bottom, while *align* = 'center' interprets these values as
the *y* coordinates of the bar centers.

The optional arguments *color*, *edgecolor*, *linewidth*,
*xerr*, and *yerr* can be either scalars or sequences of
length equal to the number of bars.  This enables you to use
barh as the basis for stacked bar charts, or candlestick
plots.

other optional kwargs:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.barh(bottom, width, height, left, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def broken_barh(xranges, yrange, hold=None, **kwargs):
    """call signature::

  broken_barh(self, xranges, yrange, **kwargs)

A collection of horizontal bars spanning *yrange* with a sequence of
*xranges*.

Required arguments:

  =========   ==============================
  Argument    Description
  =========   ==============================
  *xranges*   sequence of (*xmin*, *xwidth*)
  *yrange*    sequence of (*ymin*, *ywidth*)
  =========   ==============================

kwargs are
:class:`matplotlib.collections.BrokenBarHCollection`
properties:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

these can either be a single argument, ie::

  facecolors = 'black'

or a sequence of arguments for the various bars, ie::

  facecolors = ('black', 'red', 'green')

**Example:**

.. plot:: mpl_examples/pylab_examples/broken_barh.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.broken_barh(xranges, yrange, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def boxplot(x, notch=0, sym='b+', vert=1, whis=1.5, positions=None, widths=None, hold=None):
    """call signature::

  boxplot(x, notch=0, sym='+', vert=1, whis=1.5,
          positions=None, widths=None)

Make a box and whisker plot for each column of *x* or each
vector in sequence *x*.  The box extends from the lower to
upper quartile values of the data, with a line at the median.
The whiskers extend from the box to show the range of the
data.  Flier points are those past the end of the whiskers.

- *notch* = 0 (default) produces a rectangular box plot.
- *notch* = 1 will produce a notched box plot

*sym* (default 'b+') is the default symbol for flier points.
Enter an empty string ('') if you don't want to show fliers.

- *vert* = 1 (default) makes the boxes vertical.
- *vert* = 0 makes horizontal boxes.  This seems goofy, but
  that's how Matlab did it.

*whis* (default 1.5) defines the length of the whiskers as
a function of the inner quartile range.  They extend to the
most extreme data point within ( ``whis*(75%-25%)`` ) data range.

*positions* (default 1,2,...,n) sets the horizontal positions of
the boxes. The ticks and limits are automatically set to match
the positions.

*widths* is either a scalar or a vector and sets the width of
each box. The default is 0.5, or ``0.15*(distance between extreme
positions)`` if that is smaller.

*x* is an array or a sequence of vectors.

Returns a dictionary mapping each component of the boxplot
to a list of the :class:`matplotlib.lines.Line2D`
instances created.

**Example:**

.. plot:: pyplots/boxplot_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.boxplot(x, notch, sym, vert, whis, positions, widths)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cohere(x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=0, pad_to=None, sides='default', scale_by_freq=None, hold=None, **kwargs):
    """call signature::

  cohere(x, y, NFFT=256, Fs=2, Fc=0, detrend = mlab.detrend_none,
         window = mlab.window_hanning, noverlap=0, pad_to=None,
         sides='default', scale_by_freq=None, **kwargs)

cohere the coherence between *x* and *y*.  Coherence is the normalized
cross spectral density:

.. math::

  C_{xy} = \\frac{|P_{xy}|^2}{P_{xx}P_{yy}}

Keyword arguments:

  *NFFT*: integer
      The number of data points used in each block for the FFT.
      Must be even; a power 2 is most efficient.  The default value is 256.

  *Fs*: scalar
      The sampling frequency (samples per time unit).  It is used
      to calculate the Fourier frequencies, freqs, in cycles per time
      unit. The default value is 2.

  *detrend*: callable
      The function applied to each segment before fft-ing,
      designed to remove the mean or linear trend.  Unlike in
      matlab, where the *detrend* parameter is a vector, in
      matplotlib is it a function.  The :mod:`~matplotlib.pylab`
      module defines :func:`~matplotlib.pylab.detrend_none`,
      :func:`~matplotlib.pylab.detrend_mean`, and
      :func:`~matplotlib.pylab.detrend_linear`, but you can use
      a custom function as well.

  *window*: callable or ndarray
      A function or a vector of length *NFFT*. To create window
      vectors see :func:`window_hanning`, :func:`window_none`,
      :func:`numpy.blackman`, :func:`numpy.hamming`,
      :func:`numpy.bartlett`, :func:`scipy.signal`,
      :func:`scipy.signal.get_window`, etc. The default is
      :func:`window_hanning`.  If a function is passed as the
      argument, it must take a data segment as an argument and
      return the windowed version of the segment.

  *noverlap*: integer
      The number of points of overlap between blocks.  The default value
      is 0 (no overlap).

  *pad_to*: integer
      The number of points to which the data segment is padded when
      performing the FFT.  This can be different from *NFFT*, which
      specifies the number of data points used.  While not increasing
      the actual resolution of the psd (the minimum distance between
      resolvable peaks), this can give more points in the plot,
      allowing for more detail. This corresponds to the *n* parameter
      in the call to fft(). The default is None, which sets *pad_to*
      equal to *NFFT*

  *sides*: [ 'default' | 'onesided' | 'twosided' ]
      Specifies which sides of the PSD to return.  Default gives the
      default behavior, which returns one-sided for real data and both
      for complex data.  'onesided' forces the return of a one-sided PSD,
      while 'twosided' forces two-sided.

  *scale_by_freq*: boolean
      Specifies whether the resulting density values should be scaled
      by the scaling frequency, which gives density in units of Hz^-1.
      This allows for integration over the returned frequency values.
      The default is True for MatLab compatibility.

  *Fc*: integer
    The center frequency of *x* (defaults to 0), which offsets
    the x extents of the plot to reflect the frequency range used
    when a signal is acquired and then filtered and downsampled to
    baseband.

The return value is a tuple (*Cxy*, *f*), where *f* are the
frequencies of the coherence vector.

kwargs are applied to the lines.

References:

  * Bendat & Piersol -- Random Data: Analysis and Measurement
    Procedures, John Wiley & Sons (1986)

kwargs control the :class:`~matplotlib.lines.Line2D`
properties of the coherence plot:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/cohere_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.cohere(x, y, NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def clabel(CS, *args, **kwargs):
    """call signature::

  clabel(cs, **kwargs)

adds labels to line contours in *cs*, where *cs* is a
:class:`~matplotlib.contour.ContourSet` object returned by
contour.

::

  clabel(cs, v, **kwargs)

only labels contours listed in *v*.

Optional keyword arguments:

  *fontsize*:
    See http://matplotlib.sf.net/fonts.html

  *colors*:
    - if *None*, the color of each label matches the color of
      the corresponding contour

    - if one string color, e.g. *colors* = 'r' or *colors* =
      'red', all labels will be plotted in this color

    - if a tuple of matplotlib color args (string, float, rgb, etc),
      different labels will be plotted in different colors in the order
      specified

  *inline*:
    controls whether the underlying contour is removed or
    not. Default is *True*.

  *inline_spacing*:
    space in pixels to leave on each side of label when
    placing inline.  Defaults to 5.  This spacing will be
    exact for labels at locations where the contour is
    straight, less so for labels on curved contours.

  *fmt*:
    a format string for the label. Default is '%1.3f'
    Alternatively, this can be a dictionary matching contour
    levels with arbitrary strings to use for each contour level
    (i.e., fmt[level]=string)

  *manual*:
    if *True*, contour labels will be placed manually using
    mouse clicks.  Click the first button near a contour to
    add a label, click the second button (or potentially both
    mouse buttons at once) to finish adding labels.  The third
    button can be used to remove the last label added, but
    only if labels are not inline.  Alternatively, the keyboard
    can be used to select label locations (enter to end label
    placement, delete or backspace act like the third mouse button,
    and any other key will select a label location).

  *rightside_up*:
    if *True* (default), label rotations will always be plus
    or minus 90 degrees from level.

.. plot:: mpl_examples/pylab_examples/contour_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.clabel(CS, *args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def contour(*args, **kwargs):
    """:func:`~matplotlib.pyplot.contour` and
:func:`~matplotlib.pyplot.contourf` draw contour lines and
filled contours, respectively.  Except as noted, function
signatures and return values are the same for both versions.

:func:`~matplotlib.pyplot.contourf` differs from the Matlab
(TM) version in that it does not draw the polygon edges,
because the contouring engine yields simply connected regions
with branch cuts.  To draw the edges, add line contours with
calls to :func:`~matplotlib.pyplot.contour`.


call signatures::

  contour(Z)

make a contour plot of an array *Z*. The level values are chosen
automatically.

::

  contour(X,Y,Z)

*X*, *Y* specify the (*x*, *y*) coordinates of the surface

::

  contour(Z,N)
  contour(X,Y,Z,N)

contour *N* automatically-chosen levels.

::

  contour(Z,V)
  contour(X,Y,Z,V)

draw contour lines at the values specified in sequence *V*

::

  contourf(..., V)

fill the (len(*V*)-1) regions between the values in *V*

::

  contour(Z, **kwargs)

Use keyword args to control colors, linewidth, origin, cmap ... see
below for more details.

*X*, *Y*, and *Z* must be arrays with the same dimensions.

*Z* may be a masked array, but filled contouring may not
handle internal masked regions correctly.

``C = contour(...)`` returns a
:class:`~matplotlib.contour.ContourSet` object.

Optional keyword arguments:

  *colors*: [ None | string | (mpl_colors) ]
    If *None*, the colormap specified by cmap will be used.

    If a string, like 'r' or 'red', all levels will be plotted in this
    color.

    If a tuple of matplotlib color args (string, float, rgb, etc),
    different levels will be plotted in different colors in the order
    specified.

  *alpha*: float
    The alpha blending value

  *cmap*: [ None | Colormap ]
    A cm :class:`~matplotlib.cm.Colormap` instance or
    *None*. If *cmap* is *None* and *colors* is *None*, a
    default Colormap is used.

  *norm*: [ None | Normalize ]
    A :class:`matplotlib.colors.Normalize` instance for
    scaling data values to colors. If *norm* is *None* and
    *colors* is *None*, the default linear scaling is used.

  *origin*: [ None | 'upper' | 'lower' | 'image' ]
    If *None*, the first value of *Z* will correspond to the
    lower left corner, location (0,0). If 'image', the rc
    value for ``image.origin`` will be used.

    This keyword is not active if *X* and *Y* are specified in
    the call to contour.

  *extent*: [ None | (x0,x1,y0,y1) ]

    If *origin* is not *None*, then *extent* is interpreted as
    in :func:`matplotlib.pyplot.imshow`: it gives the outer
    pixel boundaries. In this case, the position of Z[0,0]
    is the center of the pixel, not a corner. If *origin* is
    *None*, then (*x0*, *y0*) is the position of Z[0,0], and
    (*x1*, *y1*) is the position of Z[-1,-1].

    This keyword is not active if *X* and *Y* are specified in
    the call to contour.

  *locator*: [ None | ticker.Locator subclass ]
    If *locator* is None, the default
    :class:`~matplotlib.ticker.MaxNLocator` is used. The
    locator is used to determine the contour levels if they
    are not given explicitly via the *V* argument.

  *extend*: [ 'neither' | 'both' | 'min' | 'max' ]
    Unless this is 'neither', contour levels are automatically
    added to one or both ends of the range so that all data
    are included. These added ranges are then mapped to the
    special colormap values which default to the ends of the
    colormap range, but can be set via
    :meth:`matplotlib.cm.Colormap.set_under` and
    :meth:`matplotlib.cm.Colormap.set_over` methods.

contour-only keyword arguments:

  *linewidths*: [ None | number | tuple of numbers ]
    If *linewidths* is *None*, the default width in
    ``lines.linewidth`` in ``matplotlibrc`` is used.

    If a number, all levels will be plotted with this linewidth.

    If a tuple, different levels will be plotted with different
    linewidths in the order specified

  *linestyles*: [None | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
    If *linestyles* is *None*, the 'solid' is used.

    *linestyles* can also be an iterable of the above strings
    specifying a set of linestyles to be used. If this
    iterable is shorter than the number of contour levels
    it will be repeated as necessary.

    If contour is using a monochrome colormap and the contour
    level is less than 0, then the linestyle specified
    in ``contour.negative_linestyle`` in ``matplotlibrc``
    will be used.

contourf-only keyword arguments:

  *antialiased*: [ True | False ]
    enable antialiasing

  *nchunk*: [ 0 | integer ]
    If 0, no subdivision of the domain. Specify a positive integer to
    divide the domain into subdomains of roughly *nchunk* by *nchunk*
    points. This may never actually be advantageous, so this option may
    be removed. Chunking introduces artifacts at the chunk boundaries
    unless *antialiased* is *False*.

**Example:**

.. plot:: mpl_examples/pylab_examples/contour_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.contour(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    if ret._A is not None: gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def contourf(*args, **kwargs):
    """:func:`~matplotlib.pyplot.contour` and
:func:`~matplotlib.pyplot.contourf` draw contour lines and
filled contours, respectively.  Except as noted, function
signatures and return values are the same for both versions.

:func:`~matplotlib.pyplot.contourf` differs from the Matlab
(TM) version in that it does not draw the polygon edges,
because the contouring engine yields simply connected regions
with branch cuts.  To draw the edges, add line contours with
calls to :func:`~matplotlib.pyplot.contour`.


call signatures::

  contour(Z)

make a contour plot of an array *Z*. The level values are chosen
automatically.

::

  contour(X,Y,Z)

*X*, *Y* specify the (*x*, *y*) coordinates of the surface

::

  contour(Z,N)
  contour(X,Y,Z,N)

contour *N* automatically-chosen levels.

::

  contour(Z,V)
  contour(X,Y,Z,V)

draw contour lines at the values specified in sequence *V*

::

  contourf(..., V)

fill the (len(*V*)-1) regions between the values in *V*

::

  contour(Z, **kwargs)

Use keyword args to control colors, linewidth, origin, cmap ... see
below for more details.

*X*, *Y*, and *Z* must be arrays with the same dimensions.

*Z* may be a masked array, but filled contouring may not
handle internal masked regions correctly.

``C = contour(...)`` returns a
:class:`~matplotlib.contour.ContourSet` object.

Optional keyword arguments:

  *colors*: [ None | string | (mpl_colors) ]
    If *None*, the colormap specified by cmap will be used.

    If a string, like 'r' or 'red', all levels will be plotted in this
    color.

    If a tuple of matplotlib color args (string, float, rgb, etc),
    different levels will be plotted in different colors in the order
    specified.

  *alpha*: float
    The alpha blending value

  *cmap*: [ None | Colormap ]
    A cm :class:`~matplotlib.cm.Colormap` instance or
    *None*. If *cmap* is *None* and *colors* is *None*, a
    default Colormap is used.

  *norm*: [ None | Normalize ]
    A :class:`matplotlib.colors.Normalize` instance for
    scaling data values to colors. If *norm* is *None* and
    *colors* is *None*, the default linear scaling is used.

  *origin*: [ None | 'upper' | 'lower' | 'image' ]
    If *None*, the first value of *Z* will correspond to the
    lower left corner, location (0,0). If 'image', the rc
    value for ``image.origin`` will be used.

    This keyword is not active if *X* and *Y* are specified in
    the call to contour.

  *extent*: [ None | (x0,x1,y0,y1) ]

    If *origin* is not *None*, then *extent* is interpreted as
    in :func:`matplotlib.pyplot.imshow`: it gives the outer
    pixel boundaries. In this case, the position of Z[0,0]
    is the center of the pixel, not a corner. If *origin* is
    *None*, then (*x0*, *y0*) is the position of Z[0,0], and
    (*x1*, *y1*) is the position of Z[-1,-1].

    This keyword is not active if *X* and *Y* are specified in
    the call to contour.

  *locator*: [ None | ticker.Locator subclass ]
    If *locator* is None, the default
    :class:`~matplotlib.ticker.MaxNLocator` is used. The
    locator is used to determine the contour levels if they
    are not given explicitly via the *V* argument.

  *extend*: [ 'neither' | 'both' | 'min' | 'max' ]
    Unless this is 'neither', contour levels are automatically
    added to one or both ends of the range so that all data
    are included. These added ranges are then mapped to the
    special colormap values which default to the ends of the
    colormap range, but can be set via
    :meth:`matplotlib.cm.Colormap.set_under` and
    :meth:`matplotlib.cm.Colormap.set_over` methods.

contour-only keyword arguments:

  *linewidths*: [ None | number | tuple of numbers ]
    If *linewidths* is *None*, the default width in
    ``lines.linewidth`` in ``matplotlibrc`` is used.

    If a number, all levels will be plotted with this linewidth.

    If a tuple, different levels will be plotted with different
    linewidths in the order specified

  *linestyles*: [None | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
    If *linestyles* is *None*, the 'solid' is used.

    *linestyles* can also be an iterable of the above strings
    specifying a set of linestyles to be used. If this
    iterable is shorter than the number of contour levels
    it will be repeated as necessary.

    If contour is using a monochrome colormap and the contour
    level is less than 0, then the linestyle specified
    in ``contour.negative_linestyle`` in ``matplotlibrc``
    will be used.

contourf-only keyword arguments:

  *antialiased*: [ True | False ]
    enable antialiasing

  *nchunk*: [ 0 | integer ]
    If 0, no subdivision of the domain. Specify a positive integer to
    divide the domain into subdomains of roughly *nchunk* by *nchunk*
    points. This may never actually be advantageous, so this option may
    be removed. Chunking introduces artifacts at the chunk boundaries
    unless *antialiased* is *False*.

**Example:**

.. plot:: mpl_examples/pylab_examples/contour_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.contourf(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    if ret._A is not None: gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def csd(x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=0, pad_to=None, sides='default', scale_by_freq=None, hold=None, **kwargs):
    """call signature::

  csd(x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
      window=mlab.window_hanning, noverlap=0, pad_to=None,
      sides='default', scale_by_freq=None, **kwargs)

The cross spectral density :math:`P_{xy}` by Welch's average
periodogram method.  The vectors *x* and *y* are divided into
*NFFT* length segments.  Each segment is detrended by function
*detrend* and windowed by function *window*.  The product of
the direct FFTs of *x* and *y* are averaged over each segment
to compute :math:`P_{xy}`, with a scaling to correct for power
loss due to windowing.

Returns the tuple (*Pxy*, *freqs*).  *P* is the cross spectrum
(complex valued), and :math:`10\\log_{10}|P_{xy}|` is
plotted.

Keyword arguments:

  *NFFT*: integer
      The number of data points used in each block for the FFT.
      Must be even; a power 2 is most efficient.  The default value is 256.

  *Fs*: scalar
      The sampling frequency (samples per time unit).  It is used
      to calculate the Fourier frequencies, freqs, in cycles per time
      unit. The default value is 2.

  *detrend*: callable
      The function applied to each segment before fft-ing,
      designed to remove the mean or linear trend.  Unlike in
      matlab, where the *detrend* parameter is a vector, in
      matplotlib is it a function.  The :mod:`~matplotlib.pylab`
      module defines :func:`~matplotlib.pylab.detrend_none`,
      :func:`~matplotlib.pylab.detrend_mean`, and
      :func:`~matplotlib.pylab.detrend_linear`, but you can use
      a custom function as well.

  *window*: callable or ndarray
      A function or a vector of length *NFFT*. To create window
      vectors see :func:`window_hanning`, :func:`window_none`,
      :func:`numpy.blackman`, :func:`numpy.hamming`,
      :func:`numpy.bartlett`, :func:`scipy.signal`,
      :func:`scipy.signal.get_window`, etc. The default is
      :func:`window_hanning`.  If a function is passed as the
      argument, it must take a data segment as an argument and
      return the windowed version of the segment.

  *noverlap*: integer
      The number of points of overlap between blocks.  The default value
      is 0 (no overlap).

  *pad_to*: integer
      The number of points to which the data segment is padded when
      performing the FFT.  This can be different from *NFFT*, which
      specifies the number of data points used.  While not increasing
      the actual resolution of the psd (the minimum distance between
      resolvable peaks), this can give more points in the plot,
      allowing for more detail. This corresponds to the *n* parameter
      in the call to fft(). The default is None, which sets *pad_to*
      equal to *NFFT*

  *sides*: [ 'default' | 'onesided' | 'twosided' ]
      Specifies which sides of the PSD to return.  Default gives the
      default behavior, which returns one-sided for real data and both
      for complex data.  'onesided' forces the return of a one-sided PSD,
      while 'twosided' forces two-sided.

  *scale_by_freq*: boolean
      Specifies whether the resulting density values should be scaled
      by the scaling frequency, which gives density in units of Hz^-1.
      This allows for integration over the returned frequency values.
      The default is True for MatLab compatibility.

  *Fc*: integer
    The center frequency of *x* (defaults to 0), which offsets
    the x extents of the plot to reflect the frequency range used
    when a signal is acquired and then filtered and downsampled to
    baseband.

References:
  Bendat & Piersol -- Random Data: Analysis and Measurement
  Procedures, John Wiley & Sons (1986)

kwargs control the Line2D properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/csd_demo.py

.. seealso:

    :meth:`psd`
        For a description of the optional parameters.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.csd(x, y, NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def errorbar(x, y, yerr=None, xerr=None, fmt='-', ecolor=None, elinewidth=None, capsize=3, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, hold=None, **kwargs):
    """call signature::

  errorbar(x, y, yerr=None, xerr=None,
           fmt='-', ecolor=None, elinewidth=None, capsize=3,
           barsabove=False, lolims=False, uplims=False,
           xlolims=False, xuplims=False)

Plot *x* versus *y* with error deltas in *yerr* and *xerr*.
Vertical errorbars are plotted if *yerr* is not *None*.
Horizontal errorbars are plotted if *xerr* is not *None*.

*x*, *y*, *xerr*, and *yerr* can all be scalars, which plots a
single error bar at *x*, *y*.

Optional keyword arguments:

  *xerr*/*yerr*: [ scalar | N, Nx1, or 2xN array-like ]
    If a scalar number, len(N) array-like object, or an Nx1 array-like
    object, errorbars are drawn +/- value.

    If a rank-1, 2xN numpy array, errorbars are drawn at -row1 and
    +row2

  *fmt*: '-'
    The plot format symbol for *y*. If *fmt* is *None*, just plot the
    errorbars with no line symbols.  This can be useful for creating a
    bar plot with errorbars.

  *ecolor*: [ None | mpl color ]
    a matplotlib color arg which gives the color the errorbar lines; if
    *None*, use the marker color.

  *elinewidth*: scalar
    the linewidth of the errorbar lines. If *None*, use the linewidth.

  *capsize*: scalar
    the size of the error bar caps in points

  *barsabove*: [ True | False ]
    if *True*, will plot the errorbars above the plot
    symbols. Default is below.

  *lolims*/*uplims*/*xlolims*/*xuplims*: [ False | True ]
    These arguments can be used to indicate that a value gives
    only upper/lower limits. In that case a caret symbol is
    used to indicate this. lims-arguments may be of the same
    type as *xerr* and *yerr*.

All other keyword arguments are passed on to the plot command for the
markers, so you can add additional key=value pairs to control the
errorbar markers.  For example, this code makes big red squares with
thick green edges::

  x,y,yerr = rand(3,10)
  errorbar(x, y, yerr, marker='s',
           mfc='red', mec='green', ms=20, mew=4)

where *mfc*, *mec*, *ms* and *mew* are aliases for the longer
property names, *markerfacecolor*, *markeredgecolor*, *markersize*
and *markeredgewith*.

valid kwargs for the marker properties are

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

Return value is a length 3 tuple.  The first element is the
:class:`~matplotlib.lines.Line2D` instance for the *y* symbol
lines.  The second element is a list of error bar cap lines,
the third element is a list of
:class:`~matplotlib.collections.LineCollection` instances for
the horizontal and vertical error ranges.

**Example:**

.. plot:: mpl_examples/pylab_examples/errorbar_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.errorbar(x, y, yerr, xerr, fmt, ecolor, elinewidth, capsize, barsabove, lolims, uplims, xlolims, xuplims, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def fill(*args, **kwargs):
    """call signature::

  fill(*args, **kwargs)

Plot filled polygons.  *args* is a variable length argument,
allowing for multiple *x*, *y* pairs with an optional color
format string; see :func:`~matplotlib.pyplot.plot` for details
on the argument parsing.  For example, to plot a polygon with
vertices at *x*, *y* in blue.::

  ax.fill(x,y, 'b' )

An arbitrary number of *x*, *y*, *color* groups can be specified::

  ax.fill(x1, y1, 'g', x2, y2, 'r')

Return value is a list of :class:`~matplotlib.patches.Patch`
instances that were added.

The same color strings that :func:`~matplotlib.pyplot.plot`
supports are supported by the fill format string.

If you would like to fill below a curve, eg. shade a region
between 0 and *y* along *x*, use :meth:`fill_between`

The *closed* kwarg will close the polygon when *True* (default).

kwargs control the Polygon properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/fill_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.fill(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def fill_between(x, y1, y2=0, where=None, hold=None, **kwargs):
    """call signature::

  fill_between(x, y1, y2=0, where=None, **kwargs)

Create a :class:`~matplotlib.collections.PolyCollection`
filling the regions between *y1* and *y2* where
``where==True``

*x*
  an N length np array of the x data

*y1*
  an N length scalar or np array of the y data

*y2*
  an N length scalar or np array of the y data

*where*
   if None, default to fill between everywhere.  If not None,
   it is a a N length numpy boolean array and the fill will
   only happen over the regions where ``where==True``

*kwargs*
  keyword args passed on to the :class:`PolyCollection`

kwargs control the Polygon properties:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

.. plot:: mpl_examples/pylab_examples/fill_between_demo.py

.. seealso::

    :meth:`fill_betweenx`
        for filling between two sets of x-values

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.fill_between(x, y1, y2, where, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def fill_betweenx(y, x1, x2=0, where=None, hold=None, **kwargs):
    """call signature::

  fill_between(y, x1, x2=0, where=None, **kwargs)

Create a :class:`~matplotlib.collections.PolyCollection`
filling the regions between *x1* and *x2* where
``where==True``

*y*
  an N length np array of the y data

*x1*
  an N length scalar or np array of the x data

*x2*
  an N length scalar or np array of the x data

*where*
   if None, default to fill between everywhere.  If not None,
   it is a a N length numpy boolean array and the fill will
   only happen over the regions where ``where==True``

*kwargs*
  keyword args passed on to the :class:`PolyCollection`

kwargs control the Polygon properties:

%(PolyCollection)s

.. plot:: mpl_examples/pylab_examples/fill_betweenx_demo.py

.. seealso::

    :meth:`fill_between`
        for filling between two sets of y-values

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.fill_betweenx(y, x1, x2, where, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hexbin(x, y, C=None, gridsize=100, bins=None, xscale='linear', yscale='linear', extent=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=1.0, linewidths=None, edgecolors='none', reduce_C_function=np.mean, mincnt=None, marginals=False, hold=None, **kwargs):
    """call signature::

  hexbin(x, y, C = None, gridsize = 100, bins = None,
         xscale = 'linear', yscale = 'linear',
         cmap=None, norm=None, vmin=None, vmax=None,
         alpha=1.0, linewidths=None, edgecolors='none'
         reduce_C_function = np.mean, mincnt=None, marginals=True
         **kwargs)

Make a hexagonal binning plot of *x* versus *y*, where *x*,
*y* are 1-D sequences of the same length, *N*. If *C* is None
(the default), this is a histogram of the number of occurences
of the observations at (x[i],y[i]).

If *C* is specified, it specifies values at the coordinate
(x[i],y[i]). These values are accumulated for each hexagonal
bin and then reduced according to *reduce_C_function*, which
defaults to numpy's mean function (np.mean). (If *C* is
specified, it must also be a 1-D sequence of the same length
as *x* and *y*.)

*x*, *y* and/or *C* may be masked arrays, in which case only
unmasked points will be plotted.

Optional keyword arguments:

  *gridsize*: [ 100 | integer ]
    The number of hexagons in the *x*-direction, default is
    100. The corresponding number of hexagons in the
    *y*-direction is chosen such that the hexagons are
    approximately regular. Alternatively, gridsize can be a
    tuple with two elements specifying the number of hexagons
    in the *x*-direction and the *y*-direction.

  *bins*: [ None | 'log' | integer | sequence ]
    If *None*, no binning is applied; the color of each hexagon
    directly corresponds to its count value.

    If 'log', use a logarithmic scale for the color
    map. Internally, :math:`log_{10}(i+1)` is used to
    determine the hexagon color.

    If an integer, divide the counts in the specified number
    of bins, and color the hexagons accordingly.

    If a sequence of values, the values of the lower bound of
    the bins to be used.

  *xscale*: [ 'linear' | 'log' ]
    Use a linear or log10 scale on the horizontal axis.

  *scale*: [ 'linear' | 'log' ]
    Use a linear or log10 scale on the vertical axis.

  *mincnt*: None | a positive integer
    If not None, only display cells with more than *mincnt*
    number of points in the cell

  *marginals*: True|False
    if marginals is True, plot the marginal density as
    colormapped rectagles along the bottom of the x-axis and
    left of the y-axis

  *extent*: [ None | scalars (left, right, bottom, top) ]
    The limits of the bins. The default assigns the limits
    based on gridsize, x, y, xscale and yscale.

Other keyword arguments controlling color mapping and normalization
arguments:

  *cmap*: [ None | Colormap ]
    a :class:`matplotlib.cm.Colormap` instance. If *None*,
    defaults to rc ``image.cmap``.

  *norm*: [ None | Normalize ]
    :class:`matplotlib.colors.Normalize` instance is used to
    scale luminance data to 0,1.

  *vmin*/*vmax*: scalar
    *vmin* and *vmax* are used in conjunction with *norm* to normalize
    luminance data.  If either are *None*, the min and max of the color
    array *C* is used.  Note if you pass a norm instance, your settings
    for *vmin* and *vmax* will be ignored.

  *alpha*: scalar
    the alpha value for the patches

  *linewidths*: [ None | scalar ]
    If *None*, defaults to rc lines.linewidth. Note that this
    is a tuple, and if you set the linewidths argument you
    must set it as a sequence of floats, as required by
    :class:`~matplotlib.collections.RegularPolyCollection`.

Other keyword arguments controlling the Collection properties:

  *edgecolors*: [ None | mpl color | color sequence ]
    If 'none', draws the edges in the same color as the fill color.
    This is the default, as it avoids unsightly unpainted pixels
    between the hexagons.

    If *None*, draws the outlines in the default color.

    If a matplotlib color arg or sequence of rgba tuples, draws the
    outlines in the specified color.

Here are the standard descriptions of all the
:class:`~matplotlib.collections.Collection` kwargs:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

The return value is a
:class:`~matplotlib.collections.PolyCollection` instance; use
:meth:`~matplotlib.collection.PolyCollection.get_array` on
this :class:`~matplotlib.collections.PolyCollection` to get
the counts in each hexagon..  If marginals is True, horizontal
bar and vertical bar (both PolyCollections) will be attached
to the return collection as attributes *hbar* and *vbar*


**Example:**

.. plot:: mpl_examples/pylab_examples/hexbin_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.hexbin(x, y, C, gridsize, bins, xscale, yscale, extent, cmap, norm, vmin, vmax, alpha, linewidths, edgecolors, reduce_C_function, mincnt, marginals, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hist(x, bins=10, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, hold=None, **kwargs):
    """call signature::

  hist(x, bins=10, range=None, normed=False, cumulative=False,
       bottom=None, histtype='bar', align='mid',
       orientation='vertical', rwidth=None, log=False, **kwargs)

Compute and draw the histogram of *x*. The return value is a
tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*,
[*patches0*, *patches1*,...]) if the input contains multiple
data.

Keyword arguments:

  *bins*:
    Either an integer number of bins or a sequence giving the
    bins.  *x* are the data to be binned. *x* can be an array,
    a 2D array with multiple data in its columns, or a list of
    arrays with data of different length.  Note, if *bins*
    is an integer input argument=numbins, *bins* + 1 bin edges
    will be returned, compatible with the semantics of
    :func:`numpy.histogram` with the *new* = True argument.
    Unequally spaced bins are supported if *bins* is a sequence.

  *range*:
    The lower and upper range of the bins. Lower and upper outliers
    are ignored. If not provided, *range* is (x.min(), x.max()).
    Range has no effect if *bins* is a sequence.

    If *bins* is a sequence or *range* is specified, autoscaling is
    set off (*autoscale_on* is set to *False*) and the xaxis limits
    are set to encompass the full specified bin range.

  *normed*:
    If *True*, the first element of the return tuple will
    be the counts normalized to form a probability density, i.e.,
    ``n/(len(x)*dbin)``.  In a probability density, the integral of
    the histogram should be 1; you can verify that with a
    trapezoidal integration of the probability density function::

      pdf, bins, patches = ax.hist(...)
      print np.sum(pdf * np.diff(bins))

  *weights*
    An array of weights, of the same shape as *x*.  Each value in
    *x* only contributes its associated weight towards the bin
    count (instead of 1).  If *normed* is True, the weights are
    normalized, so that the integral of the density over the range
    remains 1.

  *cumulative*:
    If *True*, then a histogram is computed where each bin
    gives the counts in that bin plus all bins for smaller values.
    The last bin gives the total number of datapoints.  If *normed*
    is also *True* then the histogram is normalized such that the
    last bin equals 1. If *cumulative* evaluates to less than 0
    (e.g. -1), the direction of accumulation is reversed.  In this
    case, if *normed* is also *True*, then the histogram is normalized
    such that the first bin equals 1.

  *histtype*: [ 'bar' | 'barstacked' | 'step' | 'stepfilled' ]
    The type of histogram to draw.

      - 'bar' is a traditional bar-type histogram.  If multiple data
        are given the bars are aranged side by side.

      - 'barstacked' is a bar-type histogram where multiple
        data are stacked on top of each other.

      - 'step' generates a lineplot that is by default
        unfilled.

      - 'stepfilled' generates a lineplot that is by default
        filled.

  *align*: ['left' | 'mid' | 'right' ]
    Controls how the histogram is plotted.

      - 'left': bars are centered on the left bin edges.

      - 'mid': bars are centered between the bin edges.

      - 'right': bars are centered on the right bin edges.

  *orientation*: [ 'horizontal' | 'vertical' ]
    If 'horizontal', :func:`~matplotlib.pyplot.barh` will be
    used for bar-type histograms and the *bottom* kwarg will be
    the left edges.

  *rwidth*:
    The relative width of the bars as a fraction of the bin
    width.  If *None*, automatically compute the width. Ignored
    if *histtype* = 'step' or 'stepfilled'.

  *log*:
    If *True*, the histogram axis will be set to a log scale.
    If *log* is *True* and *x* is a 1D array, empty bins will
    be filtered out and only the non-empty (*n*, *bins*,
    *patches*) will be returned.

kwargs are used to update the properties of the hist
:class:`~matplotlib.patches.Rectangle` instances:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]  or None for default
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: matplotlib color arg or sequence of rgba tuples
  contains: a callable function
  edgecolor or ec: mpl color spec, or None for default, or 'none' for no color
  facecolor or fc: mpl color spec, or None for default, or 'none' for no color
  figure: a :class:`matplotlib.figure.Figure` instance
  fill: [True | False]
  gid: an id string
  hatch: [ '/' | '\\\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]
  label: any string
  linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']
  linewidth or lw: float or None for default
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

You can use labels for your histogram, and only the first
:class:`~matplotlib.patches.Rectangle` gets the label (the
others get the magic string '_nolegend_'.  This will make the
histograms work in the intuitive way for bar charts::

    ax.hist(10+2*np.random.randn(1000), label='men')
    ax.hist(12+3*np.random.randn(1000), label='women', alpha=0.5)
    ax.legend()

label can also be a sequence of strings. If multiple data is
provided in *x*, the labels are asigned sequentially to the
histograms.

**Example:**

.. plot:: mpl_examples/pylab_examples/histogram_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.hist(x, bins, range, normed, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hlines(y, xmin, xmax, colors='k', linestyles='solid', label='', hold=None, **kwargs):
    """call signature::

  hlines(y, xmin, xmax, colors='k', linestyles='solid', **kwargs)

Plot horizontal lines at each *y* from *xmin* to *xmax*.

Returns the :class:`~matplotlib.collections.LineCollection`
that was added.

Required arguments:

  *y*:
    a 1-D numpy array or iterable.

  *xmin* and *xmax*:
    can be scalars or ``len(x)`` numpy arrays.  If they are
    scalars, then the respective values are constant, else the
    widths of the lines are determined by *xmin* and *xmax*.

Optional keyword arguments:

  *colors*:
    a line collections color argument, either a single color
    or a ``len(y)`` list of colors

  *linestyles*:
    [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ]

**Example:**

.. plot:: mpl_examples/pylab_examples/hline_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.hlines(y, xmin, xmax, colors, linestyles, label, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=1.0, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, hold=None, **kwargs):
    """call signature::

  imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
         alpha=1.0, vmin=None, vmax=None, origin=None, extent=None,
         **kwargs)

Display the image in *X* to current axes.  *X* may be a float
array, a uint8 array or a PIL image. If *X* is an array, *X*
can have the following shapes:

* MxN -- luminance (grayscale, float array only)
* MxNx3 -- RGB (float or uint8 array)
* MxNx4 -- RGBA (float or uint8 array)

The value for each component of MxNx3 and MxNx4 float arrays should be
in the range 0.0 to 1.0; MxN float arrays may be normalised.

An :class:`matplotlib.image.AxesImage` instance is returned.

Keyword arguments:

  *cmap*: [ None | Colormap ]
    A :class:`matplotlib.cm.Colormap` instance, eg. cm.jet.
    If *None*, default to rc ``image.cmap`` value.

    *cmap* is ignored when *X* has RGB(A) information

  *aspect*: [ None | 'auto' | 'equal' | scalar ]
    If 'auto', changes the image aspect ratio to match that of the axes

    If 'equal', and *extent* is *None*, changes the axes
    aspect ratio to match that of the image. If *extent* is
    not *None*, the axes aspect ratio is changed to match that
    of the extent.

    If *None*, default to rc ``image.aspect`` value.

  *interpolation*:

    Acceptable values are *None*, 'nearest', 'bilinear',
      'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
      'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
      'bessel', 'mitchell', 'sinc', 'lanczos',


    If *interpolation* is *None*, default to rc
    ``image.interpolation``. See also the *filternorm* and
    *filterrad* parameters

  *norm*: [ None | Normalize ]
    An :class:`matplotlib.colors.Normalize` instance; if
    *None*, default is ``normalization()``.  This scales
    luminance -> 0-1

    *norm* is only used for an MxN float array.

  *vmin*/*vmax*: [ None | scalar ]
    Used to scale a luminance image to 0-1.  If either is
    *None*, the min and max of the luminance values will be
    used.  Note if *norm* is not *None*, the settings for
    *vmin* and *vmax* will be ignored.

  *alpha*: scalar
    The alpha blending value, between 0 (transparent) and 1 (opaque)

  *origin*: [ None | 'upper' | 'lower' ]
    Place the [0,0] index of the array in the upper left or lower left
    corner of the axes. If *None*, default to rc ``image.origin``.

  *extent*: [ None | scalars (left, right, bottom, top) ]
    Data limits for the axes.  The default assigns zero-based row,
    column indices to the *x*, *y* centers of the pixels.

  *shape*: [ None | scalars (columns, rows) ]
    For raw buffer images

  *filternorm*:
    A parameter for the antigrain image resize filter.  From the
    antigrain documentation, if *filternorm* = 1, the filter normalizes
    integer values and corrects the rounding errors. It doesn't do
    anything with the source floating point values, it corrects only
    integers according to the rule of 1.0 which means that any sum of
    pixel weights must be equal to 1.0.  So, the filter function must
    produce a graph of the proper shape.

  *filterrad*:
    The filter radius for filters that have a radius
    parameter, i.e. when interpolation is one of: 'sinc',
    'lanczos' or 'blackman'

Additional kwargs are :class:`~matplotlib.artist.Artist` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  contains: a callable function
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/image_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.imshow(X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, shape, filternorm, filterrad, imlim, resample, url, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def loglog(*args, **kwargs):
    """call signature::

  loglog(*args, **kwargs)

Make a plot with log scaling on the *x* and *y* axis.

:func:`~matplotlib.pyplot.loglog` supports all the keyword
arguments of :func:`~matplotlib.pyplot.plot` and
:meth:`matplotlib.axes.Axes.set_xscale` /
:meth:`matplotlib.axes.Axes.set_yscale`.

Notable keyword arguments:

  *basex*/*basey*: scalar > 1
    base of the *x*/*y* logarithm

  *subsx*/*subsy*: [ None | sequence ]
    the location of the minor *x*/*y* ticks; *None* defaults
    to autosubs, which depend on the number of decades in the
    plot; see :meth:`matplotlib.axes.Axes.set_xscale` /
    :meth:`matplotlib.axes.Axes.set_yscale` for details

  *nonposx*/*nonposy*: ['mask' | 'clip' ]
    non-positive values in *x* or *y* can be masked as
    invalid, or clipped to a very small positive number

The remaining valid kwargs are
:class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/log_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.loglog(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pcolor(*args, **kwargs):
    """call signatures::

  pcolor(C, **kwargs)
  pcolor(X, Y, C, **kwargs)

Create a pseudocolor plot of a 2-D array.

*C* is the array of color values.

*X* and *Y*, if given, specify the (*x*, *y*) coordinates of
the colored quadrilaterals; the quadrilateral for C[i,j] has
corners at::

  (X[i,   j],   Y[i,   j]),
  (X[i,   j+1], Y[i,   j+1]),
  (X[i+1, j],   Y[i+1, j]),
  (X[i+1, j+1], Y[i+1, j+1]).

Ideally the dimensions of *X* and *Y* should be one greater
than those of *C*; if the dimensions are the same, then the
last row and column of *C* will be ignored.

Note that the the column index corresponds to the
*x*-coordinate, and the row index corresponds to *y*; for
details, see the :ref:`Grid Orientation
<axes-pcolor-grid-orientation>` section below.

If either or both of *X* and *Y* are 1-D arrays or column vectors,
they will be expanded as needed into the appropriate 2-D arrays,
making a rectangular grid.

*X*, *Y* and *C* may be masked arrays.  If either C[i, j], or one
of the vertices surrounding C[i,j] (*X* or *Y* at [i, j], [i+1, j],
[i, j+1],[i+1, j+1]) is masked, nothing is plotted.

Keyword arguments:

  *cmap*: [ None | Colormap ]
    A :class:`matplotlib.cm.Colormap` instance. If *None*, use
    rc settings.

  norm: [ None | Normalize ]
    An :class:`matplotlib.colors.Normalize` instance is used
    to scale luminance data to 0,1. If *None*, defaults to
    :func:`normalize`.

  *vmin*/*vmax*: [ None | scalar ]
    *vmin* and *vmax* are used in conjunction with *norm* to
    normalize luminance data.  If either are *None*, the min
    and max of the color array *C* is used.  If you pass a
    *norm* instance, *vmin* and *vmax* will be ignored.

  *shading*: [ 'flat' | 'faceted' ]
    If 'faceted', a black grid is drawn around each rectangle; if
    'flat', edges are not drawn. Default is 'flat', contrary to
    Matlab(TM).

    This kwarg is deprecated; please use 'edgecolors' instead:
      * shading='flat' -- edgecolors='None'
      * shading='faceted  -- edgecolors='k'

  *edgecolors*: [ None | 'None' | color | color sequence]
    If *None*, the rc setting is used by default.

    If 'None', edges will not be visible.

    An mpl color or sequence of colors will set the edge color

  *alpha*: 0 <= scalar <= 1
    the alpha blending value

Return value is a :class:`matplotlib.collection.Collection`
instance.

.. _axes-pcolor-grid-orientation:

The grid orientation follows the Matlab(TM) convention: an
array *C* with shape (*nrows*, *ncolumns*) is plotted with
the column number as *X* and the row number as *Y*, increasing
up; hence it is plotted the way the array would be printed,
except that the *Y* axis is reversed.  That is, *C* is taken
as *C*(*y*, *x*).

Similarly for :func:`~matplotlib.pyplot.meshgrid`::

  x = np.arange(5)
  y = np.arange(3)
  X, Y = meshgrid(x,y)

is equivalent to:

  X = array([[0, 1, 2, 3, 4],
             [0, 1, 2, 3, 4],
             [0, 1, 2, 3, 4]])

  Y = array([[0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2]])

so if you have::

  C = rand( len(x), len(y))

then you need::

  pcolor(X, Y, C.T)

or::

  pcolor(C.T)

Matlab :func:`pcolor` always discards the last row and column
of *C*, but matplotlib displays the last row and column if *X* and
*Y* are not specified, or if *X* and *Y* have one more row and
column than *C*.

kwargs can be used to control the
:class:`~matplotlib.collection.PolyCollection` properties:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.pcolor(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pcolormesh(*args, **kwargs):
    """call signatures::

  pcolormesh(C)
  pcolormesh(X, Y, C)
  pcolormesh(C, **kwargs)

*C* may be a masked array, but *X* and *Y* may not.  Masked
array support is implemented via *cmap* and *norm*; in
contrast, :func:`~matplotlib.pyplot.pcolor` simply does not
draw quadrilaterals with masked colors or vertices.

Keyword arguments:

  *cmap*: [ None | Colormap ]
    A :class:`matplotlib.cm.Colormap` instance. If None, use
    rc settings.

  *norm*: [ None | Normalize ]
    A :class:`matplotlib.colors.Normalize` instance is used to
    scale luminance data to 0,1. If None, defaults to
    :func:`normalize`.

  *vmin*/*vmax*: [ None | scalar ]
    *vmin* and *vmax* are used in conjunction with *norm* to
    normalize luminance data.  If either are *None*, the min
    and max of the color array *C* is used.  If you pass a
    *norm* instance, *vmin* and *vmax* will be ignored.

  *shading*: [ 'flat' | 'faceted' ]
    If 'faceted', a black grid is drawn around each rectangle; if
    'flat', edges are not drawn. Default is 'flat', contrary to
    Matlab(TM).

    This kwarg is deprecated; please use 'edgecolors' instead:
      * shading='flat' -- edgecolors='None'
      * shading='faceted  -- edgecolors='k'

  *edgecolors*: [ None | 'None' | color | color sequence]
    If None, the rc setting is used by default.

    If 'None', edges will not be visible.

    An mpl color or sequence of colors will set the edge color

  *alpha*: 0 <= scalar <= 1
    the alpha blending value

Return value is a :class:`matplotlib.collection.QuadMesh`
object.

kwargs can be used to control the
:class:`matplotlib.collections.QuadMesh`
properties:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

.. seealso::

    :func:`~matplotlib.pyplot.pcolor`
        For an explanation of the grid orientation and the
        expansion of 1-D *X* and/or *Y* to 2-D arrays.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.pcolormesh(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.59999999999999998, shadow=False, labeldistance=1.1000000000000001, hold=None):
    """call signature::

  pie(x, explode=None, labels=None,
      colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'),
      autopct=None, pctdistance=0.6, labeldistance=1.1, shadow=False)

Make a pie chart of array *x*.  The fractional area of each
wedge is given by x/sum(x).  If sum(x) <= 1, then the values
of x give the fractional area directly and the array will not
be normalized.

Keyword arguments:

  *explode*: [ None | len(x) sequence ]
    If not *None*, is a len(*x*) array which specifies the
    fraction of the radius with which to offset each wedge.

  *colors*: [ None | color sequence ]
    A sequence of matplotlib color args through which the pie chart
    will cycle.

  *labels*: [ None | len(x) sequence of strings ]
    A sequence of strings providing the labels for each wedge

  *autopct*: [ None | format string | format function ]
    If not *None*, is a string or function used to label the
    wedges with their numeric value.  The label will be placed inside
    the wedge.  If it is a format string, the label will be ``fmt%pct``.
    If it is a function, it will be called.

  *pctdistance*: scalar
    The ratio between the center of each pie slice and the
    start of the text generated by *autopct*.  Ignored if
    *autopct* is *None*; default is 0.6.

  *labeldistance*: scalar
    The radial distance at which the pie labels are drawn

  *shadow*: [ False | True ]
    Draw a shadow beneath the pie.

The pie chart will probably look best if the figure and axes are
square.  Eg.::

  figure(figsize=(8,8))
  ax = axes([0.1, 0.1, 0.8, 0.8])

Return value:
  If *autopct* is None, return the tuple (*patches*, *texts*):

    - *patches* is a sequence of
      :class:`matplotlib.patches.Wedge` instances

    - *texts* is a list of the label
      :class:`matplotlib.text.Text` instances.

  If *autopct* is not *None*, return the tuple (*patches*,
  *texts*, *autotexts*), where *patches* and *texts* are as
  above, and *autotexts* is a list of
  :class:`~matplotlib.text.Text` instances for the numeric
  labels.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.pie(x, explode, labels, colors, autopct, pctdistance, shadow, labeldistance)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def plot(*args, **kwargs):
    """Plot lines and/or markers to the
:class:`~matplotlib.axes.Axes`.  *args* is a variable length
argument, allowing for multiple *x*, *y* pairs with an
optional format string.  For example, each of the following is
legal::

    plot(x, y)         # plot x and y using default line style and color
    plot(x, y, 'bo')   # plot x and y using blue circle markers
    plot(y)            # plot y using x as index array 0..N-1
    plot(y, 'r+')      # ditto, but with red plusses

If *x* and/or *y* is 2-dimensional, then the corresponding columns
will be plotted.

An arbitrary number of *x*, *y*, *fmt* groups can be
specified, as in::

    a.plot(x1, y1, 'g^', x2, y2, 'g-')

Return value is a list of lines that were added.

The following format string characters are accepted to control
the line style or marker:

================    ===============================
character           description
================    ===============================
``'-'``             solid line style
``'--'``            dashed line style
``'-.'``            dash-dot line style
``':'``             dotted line style
``'.'``             point marker
``','``             pixel marker
``'o'``             circle marker
``'v'``             triangle_down marker
``'^'``             triangle_up marker
``'<'``             triangle_left marker
``'>'``             triangle_right marker
``'1'``             tri_down marker
``'2'``             tri_up marker
``'3'``             tri_left marker
``'4'``             tri_right marker
``'s'``             square marker
``'p'``             pentagon marker
``'*'``             star marker
``'h'``             hexagon1 marker
``'H'``             hexagon2 marker
``'+'``             plus marker
``'x'``             x marker
``'D'``             diamond marker
``'d'``             thin_diamond marker
``'|'``             vline marker
``'_'``             hline marker
================    ===============================


The following color abbreviations are supported:

==========  ========
character   color
==========  ========
'b'         blue
'g'         green
'r'         red
'c'         cyan
'm'         magenta
'y'         yellow
'k'         black
'w'         white
==========  ========

In addition, you can specify colors in many weird and
wonderful ways, including full names (``'green'``), hex
strings (``'#008000'``), RGB or RGBA tuples (``(0,1,0,1)``) or
grayscale intensities as a string (``'0.8'``).  Of these, the
string specifications can be used in place of a ``fmt`` group,
but the tuple forms can be used only as ``kwargs``.

Line styles and colors are combined in a single format string, as in
``'bo'`` for blue circles.

The *kwargs* can be used to set line properties (any property that has
a ``set_*`` method).  You can use this to set a line label (for auto
legends), linewidth, anitialising, marker face color, etc.  Here is an
example::

    plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
    plot([1,2,3], [1,4,9], 'rs',  label='line 2')
    axis([0, 4, 0, 10])
    legend()

If you make multiple lines with one plot command, the kwargs
apply to all those lines, e.g.::

    plot(x1, y1, x2, y2, antialised=False)

Neither line will be antialiased.

You do not need to use format strings, which are just
abbreviations.  All of the line properties can be controlled
by keyword arguments.  For example, you can set the color,
marker, linestyle, and markercolor with::

    plot(x, y, color='green', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=12).  See
         :class:`~matplotlib.lines.Line2D` for details.

The kwargs are :class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

kwargs *scalex* and *scaley*, if defined, are passed on to
:meth:`~matplotlib.axes.Axes.autoscale_view` to determine
whether the *x* and *y* axes are autoscaled; the default is
*True*.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.plot(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def plot_date(x, y, fmt='bo', tz=None, xdate=True, ydate=False, hold=None, **kwargs):
    """call signature::

  plot_date(x, y, fmt='bo', tz=None, xdate=True, ydate=False, **kwargs)

Similar to the :func:`~matplotlib.pyplot.plot` command, except
the *x* or *y* (or both) data is considered to be dates, and the
axis is labeled accordingly.

*x* and/or *y* can be a sequence of dates represented as float
days since 0001-01-01 UTC.

Keyword arguments:

  *fmt*: string
    The plot format string.

  *tz*: [ None | timezone string ]
    The time zone to use in labeling dates. If *None*, defaults to rc
    value.

  *xdate*: [ True | False ]
    If *True*, the *x*-axis will be labeled with dates.

  *ydate*: [ False | True ]
    If *True*, the *y*-axis will be labeled with dates.

Note if you are using custom date tickers and formatters, it
may be necessary to set the formatters/locators after the call
to :meth:`plot_date` since :meth:`plot_date` will set the
default tick locator to
:class:`matplotlib.dates.AutoDateLocator` (if the tick
locator is not already set to a
:class:`matplotlib.dates.DateLocator` instance) and the
default tick formatter to
:class:`matplotlib.dates.AutoDateFormatter` (if the tick
formatter is not already set to a
:class:`matplotlib.dates.DateFormatter` instance).

Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

.. seealso::

    :mod:`~matplotlib.dates`
        for helper functions

    :func:`~matplotlib.dates.date2num`,
    :func:`~matplotlib.dates.num2date` and
    :func:`~matplotlib.dates.drange`
        for help on creating the required floating point
        dates.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.plot_date(x, y, fmt, tz, xdate, ydate, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def psd(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=0, pad_to=None, sides='default', scale_by_freq=None, hold=None, **kwargs):
    """call signature::

  psd(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
      window=mlab.window_hanning, noverlap=0, pad_to=None,
      sides='default', scale_by_freq=None, **kwargs)

The power spectral density by Welch's average periodogram
method.  The vector *x* is divided into *NFFT* length
segments.  Each segment is detrended by function *detrend* and
windowed by function *window*.  *noverlap* gives the length of
the overlap between segments.  The :math:`|\\mathrm{fft}(i)|^2`
of each segment :math:`i` are averaged to compute *Pxx*, with a
scaling to correct for power loss due to windowing.  *Fs* is the
sampling frequency.

Keyword arguments:

  *NFFT*: integer
      The number of data points used in each block for the FFT.
      Must be even; a power 2 is most efficient.  The default value is 256.

  *Fs*: scalar
      The sampling frequency (samples per time unit).  It is used
      to calculate the Fourier frequencies, freqs, in cycles per time
      unit. The default value is 2.

  *detrend*: callable
      The function applied to each segment before fft-ing,
      designed to remove the mean or linear trend.  Unlike in
      matlab, where the *detrend* parameter is a vector, in
      matplotlib is it a function.  The :mod:`~matplotlib.pylab`
      module defines :func:`~matplotlib.pylab.detrend_none`,
      :func:`~matplotlib.pylab.detrend_mean`, and
      :func:`~matplotlib.pylab.detrend_linear`, but you can use
      a custom function as well.

  *window*: callable or ndarray
      A function or a vector of length *NFFT*. To create window
      vectors see :func:`window_hanning`, :func:`window_none`,
      :func:`numpy.blackman`, :func:`numpy.hamming`,
      :func:`numpy.bartlett`, :func:`scipy.signal`,
      :func:`scipy.signal.get_window`, etc. The default is
      :func:`window_hanning`.  If a function is passed as the
      argument, it must take a data segment as an argument and
      return the windowed version of the segment.

  *noverlap*: integer
      The number of points of overlap between blocks.  The default value
      is 0 (no overlap).

  *pad_to*: integer
      The number of points to which the data segment is padded when
      performing the FFT.  This can be different from *NFFT*, which
      specifies the number of data points used.  While not increasing
      the actual resolution of the psd (the minimum distance between
      resolvable peaks), this can give more points in the plot,
      allowing for more detail. This corresponds to the *n* parameter
      in the call to fft(). The default is None, which sets *pad_to*
      equal to *NFFT*

  *sides*: [ 'default' | 'onesided' | 'twosided' ]
      Specifies which sides of the PSD to return.  Default gives the
      default behavior, which returns one-sided for real data and both
      for complex data.  'onesided' forces the return of a one-sided PSD,
      while 'twosided' forces two-sided.

  *scale_by_freq*: boolean
      Specifies whether the resulting density values should be scaled
      by the scaling frequency, which gives density in units of Hz^-1.
      This allows for integration over the returned frequency values.
      The default is True for MatLab compatibility.

  *Fc*: integer
    The center frequency of *x* (defaults to 0), which offsets
    the x extents of the plot to reflect the frequency range used
    when a signal is acquired and then filtered and downsampled to
    baseband.

Returns the tuple (*Pxx*, *freqs*).

For plotting, the power is plotted as
:math:`10\\log_{10}(P_{xx})` for decibels, though *Pxx* itself
is returned.

References:
  Bendat & Piersol -- Random Data: Analysis and Measurement
  Procedures, John Wiley & Sons (1986)

kwargs control the :class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/psd_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.psd(x, NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def quiver(*args, **kw):
    """Plot a 2-D field of arrows.

call signatures::

  quiver(U, V, **kw)
  quiver(U, V, C, **kw)
  quiver(X, Y, U, V, **kw)
  quiver(X, Y, U, V, C, **kw)

Arguments:

  *X*, *Y*:

    The x and y coordinates of the arrow locations (default is tail of
    arrow; see *pivot* kwarg)

  *U*, *V*:

    give the *x* and *y* components of the arrow vectors

  *C*:
    an optional array used to map colors to the arrows

All arguments may be 1-D or 2-D arrays or sequences. If *X* and *Y*
are absent, they will be generated as a uniform grid.  If *U* and *V*
are 2-D arrays but *X* and *Y* are 1-D, and if len(*X*) and len(*Y*)
match the column and row dimensions of *U*, then *X* and *Y* will be
expanded with :func:`numpy.meshgrid`.

*U*, *V*, *C* may be masked arrays, but masked *X*, *Y* are not
supported at present.

Keyword arguments:

  *units*: ['width' | 'height' | 'dots' | 'inches' | 'x' | 'y' ]
    arrow units; the arrow dimensions *except for length* are in
    multiples of this unit.

    * 'width' or 'height': the width or height of the axes

    * 'dots' or 'inches': pixels or inches, based on the figure dpi

    * 'x' or 'y': *X* or *Y* data units

    The arrows scale differently depending on the units.  For
    'x' or 'y', the arrows get larger as one zooms in; for other
    units, the arrow size is independent of the zoom state.  For
    'width or 'height', the arrow size increases with the width and
    height of the axes, respectively, when the the window is resized;
    for 'dots' or 'inches', resizing does not change the arrows.

   *angles*: ['uv' | 'xy' | array]
    With the default 'uv', the arrow aspect ratio is 1, so that
    if *U*==*V* the angle of the arrow on the plot is 45 degrees
    CCW from the *x*-axis.
    With 'xy', the arrow points from (x,y) to (x+u, y+v).
    Alternatively, arbitrary angles may be specified as an array
    of values in degrees, CCW from the *x*-axis.

  *scale*: [ None | float ]
    data units per arrow unit, e.g. m/s per plot width; a smaller
    scale parameter makes the arrow longer.  If *None*, a simple
    autoscaling algorithm is used, based on the average vector length
    and the number of vectors.

  *width*:
    shaft width in arrow units; default depends on choice of units,
    above, and number of vectors; a typical starting value is about
    0.005 times the width of the plot.

  *headwidth*: scalar
    head width as multiple of shaft width, default is 3

  *headlength*: scalar
    head length as multiple of shaft width, default is 5

  *headaxislength*: scalar
    head length at shaft intersection, default is 4.5

  *minshaft*: scalar
    length below which arrow scales, in units of head length. Do not
    set this to less than 1, or small arrows will look terrible!
    Default is 1

  *minlength*: scalar
    minimum length as a multiple of shaft width; if an arrow length
    is less than this, plot a dot (hexagon) of this diameter instead.
    Default is 1.

  *pivot*: [ 'tail' | 'middle' | 'tip' ]
    The part of the arrow that is at the grid point; the arrow rotates
    about this point, hence the name *pivot*.

  *color*: [ color | color sequence ]
    This is a synonym for the
    :class:`~matplotlib.collections.PolyCollection` facecolor kwarg.
    If *C* has been set, *color* has no effect.

The defaults give a slightly swept-back arrow; to make the head a
triangle, make *headaxislength* the same as *headlength*. To make the
arrow more pointed, reduce *headwidth* or increase *headlength* and
*headaxislength*. To make the head smaller relative to the shaft,
scale down all the head parameters. You will probably do best to leave
minshaft alone.

linewidths and edgecolors can be used to customize the arrow
outlines. Additional :class:`~matplotlib.collections.PolyCollection`
keyword arguments:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kw.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.quiver(*args, **kw)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def quiverkey(*args, **kw):
    """Add a key to a quiver plot.

call signature::

  quiverkey(Q, X, Y, U, label, **kw)

Arguments:

  *Q*:
    The Quiver instance returned by a call to quiver.

  *X*, *Y*:
    The location of the key; additional explanation follows.

  *U*:
    The length of the key

  *label*:
    a string with the length and units of the key

Keyword arguments:

  *coordinates* = [ 'axes' | 'figure' | 'data' | 'inches' ]
    Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
    normalized coordinate systems with 0,0 in the lower left and 1,1
    in the upper right; 'data' are the axes data coordinates (used for
    the locations of the vectors in the quiver plot itself); 'inches'
    is position in the figure in inches, with 0,0 at the lower left
    corner.

  *color*:
    overrides face and edge colors from *Q*.

  *labelpos* = [ 'N' | 'S' | 'E' | 'W' ]
    Position the label above, below, to the right, to the left of the
    arrow, respectively.

  *labelsep*:
    Distance in inches between the arrow and the label.  Default is
    0.1

  *labelcolor*:
    defaults to default :class:`~matplotlib.text.Text` color.

  *fontproperties*:
    A dictionary with keyword arguments accepted by the
    :class:`~matplotlib.font_manager.FontProperties` initializer:
    *family*, *style*, *variant*, *size*, *weight*

Any additional keyword arguments are used to override vector
properties taken from *Q*.

The positioning of the key depends on *X*, *Y*, *coordinates*, and
*labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position
of the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y*
positions the head, and if *labelpos* is 'W', *X*, *Y* positions the
tail; in either of these two cases, *X*, *Y* is somewhere in the
middle of the arrow+label key object.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kw.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.quiverkey(*args, **kw)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=1.0, linewidths=None, faceted=True, verts=None, hold=None, **kwargs):
    """call signatures::

  scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
          vmin=None, vmax=None, alpha=1.0, linewidths=None,
          verts=None, **kwargs)

Make a scatter plot of *x* versus *y*, where *x*, *y* are
converted to 1-D sequences which must be of the same length, *N*.

Keyword arguments:

  *s*:
    size in points^2.  It is a scalar or an array of the same
    length as *x* and *y*.

  *c*:
    a color. *c* can be a single color format string, or a
    sequence of color specifications of length *N*, or a
    sequence of *N* numbers to be mapped to colors using the
    *cmap* and *norm* specified via kwargs (see below). Note
    that *c* should not be a single numeric RGB or RGBA
    sequence because that is indistinguishable from an array
    of values to be colormapped.  *c* can be a 2-D array in
    which the rows are RGB or RGBA, however.

  *marker*:
    can be one of:

    =====   ==============
    Value   Description
    =====   ==============
    's'     square
    'o'     circle
    '^'     triangle up
    '>'     triangle right
    'v'     triangle down
    '<'     triangle left
    'd'     diamond
    'p'     pentagram
    'h'     hexagon
    '8'     octagon
    '+'     plus
    'x'     cross
    =====   ==============

    The marker can also be a tuple (*numsides*, *style*,
    *angle*), which will create a custom, regular symbol.

      *numsides*:
        the number of sides

      *style*:
        the style of the regular symbol:

        =====   =============================================
        Value   Description
        =====   =============================================
        0       a regular polygon
        1       a star-like symbol
        2       an asterisk
        3       a circle (*numsides* and *angle* is ignored)
        =====   =============================================

      *angle*:
        the angle of rotation of the symbol

    Finally, *marker* can be (*verts*, 0): *verts* is a
    sequence of (*x*, *y*) vertices for a custom scatter
    symbol.  Alternatively, use the kwarg combination
    *marker* = *None*, *verts* = *verts*.

Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in
which case all masks will be combined and only unmasked points
will be plotted.

Other keyword arguments: the color mapping and normalization
arguments will be used only if *c* is an array of floats.

  *cmap*: [ None | Colormap ]
    A :class:`matplotlib.colors.Colormap` instance. If *None*,
    defaults to rc ``image.cmap``. *cmap* is only used if *c*
    is an array of floats.

  *norm*: [ None | Normalize ]
    A :class:`matplotlib.colors.Normalize` instance is used to
    scale luminance data to 0, 1. If *None*, use the default
    :func:`normalize`. *norm* is only used if *c* is an array
    of floats.

  *vmin*/*vmax*:
    *vmin* and *vmax* are used in conjunction with norm to
    normalize luminance data.  If either are None, the min and
    max of the color array *C* is used.  Note if you pass a
    *norm* instance, your settings for *vmin* and *vmax* will
    be ignored.

  *alpha*: 0 <= scalar <= 1
    The alpha value for the patches

  *linewidths*: [ None | scalar | sequence ]
    If *None*, defaults to (lines.linewidth,).  Note that this
    is a tuple, and if you set the linewidths argument you
    must set it as a sequence of floats, as required by
    :class:`~matplotlib.collections.RegularPolyCollection`.

Optional kwargs control the
:class:`~matplotlib.collections.Collection` properties; in
particular:

  *edgecolors*:
    'none' to plot faces with no outlines

  *facecolors*:
    'none' to plot unfilled outlines

Here are the standard descriptions of all the
:class:`~matplotlib.collections.Collection` kwargs:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

A :class:`~matplotlib.collections.Collection` instance is
returned.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, faceted, verts, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def semilogx(*args, **kwargs):
    """call signature::

  semilogx(*args, **kwargs)

Make a plot with log scaling on the *x* axis.

:func:`semilogx` supports all the keyword arguments of
:func:`~matplotlib.pyplot.plot` and
:meth:`matplotlib.axes.Axes.set_xscale`.

Notable keyword arguments:

  *basex*: scalar > 1
    base of the *x* logarithm

  *subsx*: [ None | sequence ]
    The location of the minor xticks; *None* defaults to
    autosubs, which depend on the number of decades in the
    plot; see :meth:`~matplotlib.axes.Axes.set_xscale` for
    details.

  *nonposx*: ['mask' | 'clip' ]
    non-positive values in *x* can be masked as
    invalid, or clipped to a very small positive number

The remaining valid kwargs are
:class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

.. seealso::

    :meth:`loglog`
        For example code and figure

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.semilogx(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def semilogy(*args, **kwargs):
    """call signature::

  semilogy(*args, **kwargs)

Make a plot with log scaling on the *y* axis.

:func:`semilogy` supports all the keyword arguments of
:func:`~matplotlib.pylab.plot` and
:meth:`matplotlib.axes.Axes.set_yscale`.

Notable keyword arguments:

  *basey*: scalar > 1
    Base of the *y* logarithm

  *subsy*: [ None | sequence ]
    The location of the minor yticks; *None* defaults to
    autosubs, which depend on the number of decades in the
    plot; see :meth:`~matplotlib.axes.Axes.set_yscale` for
    details.

  *nonposy*: ['mask' | 'clip' ]
    non-positive values in *y* can be masked as
    invalid, or clipped to a very small positive number

The remaining valid kwargs are
:class:`~matplotlib.lines.Line2D` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number

.. seealso::

    :meth:`loglog`
        For example code and figure

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.semilogy(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default', scale_by_freq=None, hold=None, **kwargs):
    """call signature::

  specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
           window=mlab.window_hanning, noverlap=128,
           cmap=None, xextent=None, pad_to=None, sides='default',
           scale_by_freq=None, **kwargs)

Compute a spectrogram of data in *x*.  Data are split into
*NFFT* length segments and the PSD of each section is
computed.  The windowing function *window* is applied to each
segment, and the amount of overlap of each segment is
specified with *noverlap*.

Keyword arguments:

  *NFFT*: integer
      The number of data points used in each block for the FFT.
      Must be even; a power 2 is most efficient.  The default value is 256.

  *Fs*: scalar
      The sampling frequency (samples per time unit).  It is used
      to calculate the Fourier frequencies, freqs, in cycles per time
      unit. The default value is 2.

  *detrend*: callable
      The function applied to each segment before fft-ing,
      designed to remove the mean or linear trend.  Unlike in
      matlab, where the *detrend* parameter is a vector, in
      matplotlib is it a function.  The :mod:`~matplotlib.pylab`
      module defines :func:`~matplotlib.pylab.detrend_none`,
      :func:`~matplotlib.pylab.detrend_mean`, and
      :func:`~matplotlib.pylab.detrend_linear`, but you can use
      a custom function as well.

  *window*: callable or ndarray
      A function or a vector of length *NFFT*. To create window
      vectors see :func:`window_hanning`, :func:`window_none`,
      :func:`numpy.blackman`, :func:`numpy.hamming`,
      :func:`numpy.bartlett`, :func:`scipy.signal`,
      :func:`scipy.signal.get_window`, etc. The default is
      :func:`window_hanning`.  If a function is passed as the
      argument, it must take a data segment as an argument and
      return the windowed version of the segment.

  *noverlap*: integer
      The number of points of overlap between blocks.  The default value
      is 0 (no overlap).

  *pad_to*: integer
      The number of points to which the data segment is padded when
      performing the FFT.  This can be different from *NFFT*, which
      specifies the number of data points used.  While not increasing
      the actual resolution of the psd (the minimum distance between
      resolvable peaks), this can give more points in the plot,
      allowing for more detail. This corresponds to the *n* parameter
      in the call to fft(). The default is None, which sets *pad_to*
      equal to *NFFT*

  *sides*: [ 'default' | 'onesided' | 'twosided' ]
      Specifies which sides of the PSD to return.  Default gives the
      default behavior, which returns one-sided for real data and both
      for complex data.  'onesided' forces the return of a one-sided PSD,
      while 'twosided' forces two-sided.

  *scale_by_freq*: boolean
      Specifies whether the resulting density values should be scaled
      by the scaling frequency, which gives density in units of Hz^-1.
      This allows for integration over the returned frequency values.
      The default is True for MatLab compatibility.

  *Fc*: integer
    The center frequency of *x* (defaults to 0), which offsets
    the y extents of the plot to reflect the frequency range used
    when a signal is acquired and then filtered and downsampled to
    baseband.

  *cmap*:
    A :class:`matplotlib.cm.Colormap` instance; if *None* use
    default determined by rc

  *xextent*:
    The image extent along the x-axis. xextent = (xmin,xmax)
    The default is (0,max(bins)), where bins is the return
    value from :func:`mlab.specgram`

  *kwargs*:

    Additional kwargs are passed on to imshow which makes the
    specgram image

  Return value is (*Pxx*, *freqs*, *bins*, *im*):

  - *bins* are the time points the spectrogram is calculated over
  - *freqs* is an array of frequencies
  - *Pxx* is a len(times) x len(freqs) array of power
  - *im* is a :class:`matplotlib.image.AxesImage` instance

Note: If *x* is real (i.e. non-complex), only the positive
spectrum is shown.  If *x* is complex, both positive and
negative parts of the spectrum are shown.  This can be
overridden using the *sides* keyword argument.

**Example:**

.. plot:: mpl_examples/pylab_examples/specgram_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.specgram(x, NFFT, Fs, Fc, detrend, window, noverlap, cmap, xextent, pad_to, sides, scale_by_freq, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret[-1]
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spy(Z, precision=0, marker=None, markersize=None, aspect='equal', hold=None, **kwargs):
    """call signature::

  spy(Z, precision=0, marker=None, markersize=None,
      aspect='equal', **kwargs)

``spy(Z)`` plots the sparsity pattern of the 2-D array *Z*.

If *precision* is 0, any non-zero value will be plotted;
else, values of :math:`|Z| > precision` will be plotted.

For :class:`scipy.sparse.spmatrix` instances, there is a
special case: if *precision* is 'present', any value present in
the array will be plotted, even if it is identically zero.

The array will be plotted as it would be printed, with
the first index (row) increasing down and the second
index (column) increasing to the right.

By default aspect is 'equal', so that each array element
occupies a square space; set the aspect kwarg to 'auto'
to allow the plot to fill the plot box, or to any scalar
number to specify the aspect ratio of an array element
directly.

Two plotting styles are available: image or marker. Both
are available for full arrays, but only the marker style
works for :class:`scipy.sparse.spmatrix` instances.

If *marker* and *markersize* are *None*, an image will be
returned and any remaining kwargs are passed to
:func:`~matplotlib.pyplot.imshow`; else, a
:class:`~matplotlib.lines.Line2D` object will be returned with
the value of marker determining the marker type, and any
remaining kwargs passed to the
:meth:`~matplotlib.axes.Axes.plot` method.

If *marker* and *markersize* are *None*, useful kwargs include:

* *cmap*
* *alpha*

.. seealso::

    :func:`~matplotlib.pyplot.imshow`
       For image options.

For controlling colors, e.g. cyan background and red marks,
use::

  cmap = mcolors.ListedColormap(['c','r'])

If *marker* or *markersize* is not *None*, useful kwargs include:

* *marker*
* *markersize*
* *color*

Useful values for *marker* include:

* 's'  square (default)
* 'o'  circle
* '.'  point
* ','  pixel

.. seealso::

    :func:`~matplotlib.pyplot.plot`
       For plotting options

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.spy(Z, precision, marker, markersize, aspect, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    gci._current = ret
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-', hold=None):
    """call signature::

  stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')

A stem plot plots vertical lines (using *linefmt*) at each *x*
location from the baseline to *y*, and places a marker there
using *markerfmt*.  A horizontal line at 0 is is plotted using
*basefmt*.

Return value is a tuple (*markerline*, *stemlines*,
*baseline*).

.. seealso::

    `this document`__
       for details

    :file:`examples/pylab_examples/stem_plot.py`
       for a demo

__ http://www.mathworks.com/access/helpdesk/help/techdoc/ref/stem.html

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.stem(x, y, linefmt, markerfmt, basefmt)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def step(x, y, *args, **kwargs):
    """call signature::

  step(x, y, *args, **kwargs)

Make a step plot. Additional keyword args to :func:`step` are the same
as those for :func:`~matplotlib.pyplot.plot`.

*x* and *y* must be 1-D sequences, and it is assumed, but not checked,
that *x* is uniformly increasing.

Keyword arguments:

*where*: [ 'pre' | 'post' | 'mid'  ]
  If 'pre', the interval from x[i] to x[i+1] has level y[i+1]

  If 'post', that interval has level y[i]

  If 'mid', the jumps in *y* occur half-way between the
  *x*-values.

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.step(x, y, *args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def vlines(x, ymin, ymax, colors='k', linestyles='solid', label='', hold=None, **kwargs):
    """call signature::

  vlines(x, ymin, ymax, color='k', linestyles='solid')

Plot vertical lines at each *x* from *ymin* to *ymax*.  *ymin*
or *ymax* can be scalars or len(*x*) numpy arrays.  If they are
scalars, then the respective values are constant, else the
heights of the lines are determined by *ymin* and *ymax*.

*colors*
  a line collections color args, either a single color
  or a len(*x*) list of colors

*linestyles*

  one of [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ]

Returns the :class:`matplotlib.collections.LineCollection`
that was added.

kwargs are :class:`~matplotlib.collections.LineCollection` properties:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  segments: unknown
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  verts: unknown
  visible: [True | False]
  zorder: any number

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.vlines(x, ymin, ymax, colors, linestyles, label, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def xcorr(x, y, normed=True, detrend=mlab.detrend_none, usevlines=True, maxlags=10, hold=None, **kwargs):
    """call signature::

    def xcorr(self, x, y, normed=True, detrend=mlab.detrend_none,
      usevlines=True, maxlags=10, **kwargs):

Plot the cross correlation between *x* and *y*.  If *normed* =
*True*, normalize the data by the cross correlation at 0-th
lag.  *x* and y are detrended by the *detrend* callable
(default no normalization).  *x* and *y* must be equal length.

Data are plotted as ``plot(lags, c, **kwargs)``

Return value is a tuple (*lags*, *c*, *line*) where:

  - *lags* are a length ``2*maxlags+1`` lag vector

  - *c* is the ``2*maxlags+1`` auto correlation vector

  - *line* is a :class:`~matplotlib.lines.Line2D` instance
     returned by :func:`~matplotlib.pyplot.plot`.

The default *linestyle* is *None* and the default *marker* is
'o', though these can be overridden with keyword args.  The
cross correlation is performed with :func:`numpy.correlate`
with *mode* = 2.

If *usevlines* is *True*:

   :func:`~matplotlib.pyplot.vlines`
   rather than :func:`~matplotlib.pyplot.plot` is used to draw
   vertical lines from the origin to the xcorr.  Otherwise the
   plotstyle is determined by the kwargs, which are
   :class:`~matplotlib.lines.Line2D` properties.

   The return value is a tuple (*lags*, *c*, *linecol*, *b*)
   where *linecol* is the
   :class:`matplotlib.collections.LineCollection` instance and
   *b* is the *x*-axis.

*maxlags* is a positive integer detailing the number of lags to show.
The default value of *None* will return all ``(2*len(x)-1)`` lags.

**Example:**

:func:`~matplotlib.pyplot.xcorr` above, and
:func:`~matplotlib.pyplot.acorr` below.

**Example:**

.. plot:: mpl_examples/pylab_examples/xcorr_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.xcorr(x, y, normed, detrend, usevlines, maxlags, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def barbs(*args, **kw):
    """Plot a 2-D field of barbs.

call signatures::

  barb(U, V, **kw)
  barb(U, V, C, **kw)
  barb(X, Y, U, V, **kw)
  barb(X, Y, U, V, C, **kw)

Arguments:

  *X*, *Y*:
    The x and y coordinates of the barb locations
    (default is head of barb; see *pivot* kwarg)

  *U*, *V*:
    give the *x* and *y* components of the barb shaft

  *C*:
    an optional array used to map colors to the barbs

All arguments may be 1-D or 2-D arrays or sequences. If *X* and *Y*
are absent, they will be generated as a uniform grid.  If *U* and *V*
are 2-D arrays but *X* and *Y* are 1-D, and if len(*X*) and len(*Y*)
match the column and row dimensions of *U*, then *X* and *Y* will be
expanded with :func:`numpy.meshgrid`.

*U*, *V*, *C* may be masked arrays, but masked *X*, *Y* are not
supported at present.

Keyword arguments:

  *length*:
    Length of the barb in points; the other parts of the barb
    are scaled against this.
    Default is 9

  *pivot*: [ 'tip' | 'middle' ]
    The part of the arrow that is at the grid point; the arrow rotates
    about this point, hence the name *pivot*.  Default is 'tip'

  *barbcolor*: [ color | color sequence ]
    Specifies the color all parts of the barb except any flags.  This
    parameter is analagous to the *edgecolor* parameter for polygons,
    which can be used instead. However this parameter will override
    facecolor.

  *flagcolor*: [ color | color sequence ]
    Specifies the color of any flags on the barb.  This parameter is
    analagous to the *facecolor* parameter for polygons, which can be
    used instead. However this parameter will override facecolor.  If
    this is not set (and *C* has not either) then *flagcolor* will be
    set to match *barbcolor* so that the barb has a uniform color. If
    *C* has been set, *flagcolor* has no effect.

  *sizes*:
    A dictionary of coefficients specifying the ratio of a given
    feature to the length of the barb. Only those values one wishes to
    override need to be included.  These features include:

        - 'spacing' - space between features (flags, full/half barbs)

        - 'height' - height (distance from shaft to top) of a flag or
          full barb

        - 'width' - width of a flag, twice the width of a full barb

        - 'emptybarb' - radius of the circle used for low magnitudes

  *fill_empty*:
    A flag on whether the empty barbs (circles) that are drawn should
    be filled with the flag color.  If they are not filled, they will
    be drawn such that no color is applied to the center.  Default is
    False

  *rounding*:
    A flag to indicate whether the vector magnitude should be rounded
    when allocating barb components.  If True, the magnitude is
    rounded to the nearest multiple of the half-barb increment.  If
    False, the magnitude is simply truncated to the next lowest
    multiple.  Default is True

  *barb_increments*:
    A dictionary of increments specifying values to associate with
    different parts of the barb. Only those values one wishes to
    override need to be included.

        - 'half' - half barbs (Default is 5)

        - 'full' - full barbs (Default is 10)

        - 'flag' - flags (default is 50)

  *flip_barb*:
    Either a single boolean flag or an array of booleans.  Single
    boolean indicates whether the lines and flags should point
    opposite to normal for all barbs.  An array (which should be the
    same size as the other data arrays) indicates whether to flip for
    each individual barb.  Normal behavior is for the barbs and lines
    to point right (comes from wind barbs having these features point
    towards low pressure in the Northern Hemisphere.)  Default is
    False

Barbs are traditionally used in meteorology as a way to plot the speed
and direction of wind observations, but can technically be used to
plot any two dimensional vector quantity.  As opposed to arrows, which
give vector magnitude by the length of the arrow, the barbs give more
quantitative information about the vector magnitude by putting slanted
lines or a triangle for various increments in magnitude, as show
schematically below::

 :     /\\    \\
 :    /  \\    \\
 :   /    \\    \\    \\
 :  /      \\    \\    \\
 : ------------------------------

.. note the double \\ at the end of each line to make the figure
.. render correctly

The largest increment is given by a triangle (or "flag"). After those
come full lines (barbs). The smallest increment is a half line.  There
is only, of course, ever at most 1 half line.  If the magnitude is
small and only needs a single half-line and no full lines or
triangles, the half-line is offset from the end of the barb so that it
can be easily distinguished from barbs with a single full line.  The
magnitude for the barb shown above would nominally be 65, using the
standard increments of 50, 10, and 5.

linewidths and edgecolors can be used to customize the barb.
Additional :class:`~matplotlib.collections.PolyCollection` keyword
arguments:

  alpha: float
  animated: [True | False]
  antialiased or antialiaseds: Boolean or sequence of booleans
  array: unknown
  axes: an :class:`~matplotlib.axes.Axes` instance
  clim: a length 2 sequence of floats
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  cmap: a colormap
  color: matplotlib color arg or sequence of rgba tuples
  colorbar: unknown
  contains: a callable function
  edgecolor or edgecolors: matplotlib color arg or sequence of rgba tuples
  facecolor or facecolors: matplotlib color arg or sequence of rgba tuples
  figure: a :class:`matplotlib.figure.Figure` instance
  gid: an id string
  label: any string
  linestyle or linestyles or dashes: ['solid' | 'dashed', 'dashdot', 'dotted' |         (offset, on-off-dash-seq) ]
  linewidth or lw or linewidths: float or sequence of floats
  lod: [True | False]
  norm: unknown
  offsets: float or sequence of floats
  picker: [None|float|boolean|callable]
  pickradius: unknown
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  urls: unknown
  visible: [True | False]
  zorder: any number

**Example:**

.. plot:: mpl_examples/pylab_examples/barb_demo.py

Additional kwargs: hold = [True|False] overrides default hold state
"""
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kw.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.barbs(*args, **kw)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cla():
    """Clear the current axes"""
    ret =  gca().cla()
    draw_if_interactive()
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def grid(b=None, **kwargs):
    """call signature::

  grid(self, b=None, **kwargs)

Set the axes grids on or off; *b* is a boolean

If *b* is *None* and ``len(kwargs)==0``, toggle the grid state.  If
*kwargs* are supplied, it is assumed that you want a grid and *b*
is thus set to *True*

*kawrgs* are used to set the grid line properties, eg::

  ax.grid(color='r', linestyle='-', linewidth=2)

Valid :class:`~matplotlib.lines.Line2D` kwargs are

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  antialiased or aa: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color or c: any matplotlib color
  contains: a callable function
  dash_capstyle: ['butt' | 'round' | 'projecting']
  dash_joinstyle: ['miter' | 'round' | 'bevel']
  dashes: sequence of on/off ink in points
  data: 2D array
  drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']
  gid: an id string
  label: any string
  linestyle or ls: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and         any drawstyle in combination with a linestyle, e.g. 'steps--'.
  linewidth or lw: float value in points
  lod: [True | False]
  marker: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'                  | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'                  | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'                  | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT                  | 'None' | ' ' | '' ]
  markeredgecolor or mec: any matplotlib color
  markeredgewidth or mew: float value in points
  markerfacecolor or mfc: any matplotlib color
  markersize or ms: float
  markevery: None | integer | (startind, stride)
  picker: float distance in points or callable pick function         ``fn(artist, event)``
  pickradius: float distance in points
  rasterized: [True | False | None]
  snap: unknown
  solid_capstyle: ['butt' | 'round' |  'projecting']
  solid_joinstyle: ['miter' | 'round' | 'bevel']
  transform: a :class:`matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  xdata: 1D array
  ydata: 1D array
  zorder: any number         """
    ret =  gca().grid(b, **kwargs)
    draw_if_interactive()
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def legend(*args, **kwargs):
    """call signature::

  legend(*args, **kwargs)

Place a legend on the current axes at location *loc*.  Labels are a
sequence of strings and *loc* can be a string or an integer specifying
the legend location.

To make a legend with existing lines::

  legend()

:meth:`legend` by itself will try and build a legend using the label
property of the lines/patches/collections.  You can set the label of
a line by doing::

  plot(x, y, label='my data')

or::

  line.set_label('my data').

If label is set to '_nolegend_', the item will not be shown in
legend.

To automatically generate the legend from labels::

  legend( ('label1', 'label2', 'label3') )

To make a legend for a list of lines and labels::

  legend( (line1, line2, line3), ('label1', 'label2', 'label3') )

To make a legend at a given location, using a location argument::

  legend( ('label1', 'label2', 'label3'), loc='upper left')

or::

  legend( (line1, line2, line3),  ('label1', 'label2', 'label3'), loc=2)

The location codes are

  ===============   =============
  Location String   Location Code
  ===============   =============
  'best'            0
  'upper right'     1
  'upper left'      2
  'lower left'      3
  'lower right'     4
  'right'           5
  'center left'     6
  'center right'    7
  'lower center'    8
  'upper center'    9
  'center'          10
  ===============   =============


Users can specify any arbitrary location for the legend using the
*bbox_to_anchor* keyword argument. bbox_to_anchor can be an instance
of BboxBase(or its derivatives) or a tuple of 2 or 4 floats.
For example, ::
        
 loc = 'upper right', bbox_to_anchor = (0.5, 0.5)

will place the legend so that the upper right corner of the legend at
the center of the axes.
        
The legend location can be specified in other coordinate, by using the
*bbox_transform* keyword.

The loc itslef can be a 2-tuple giving x,y of the lower-left corner of
the legend in axes coords (*bbox_to_anchor* is ignored).


Keyword arguments:

  *isaxes*: [ True | False ]
    Indicates that this is an axes legend

  *prop*: [ None | FontProperties | dict ]
    A :class:`matplotlib.font_manager.FontProperties`
    instance. If *prop* is a dictionary, a new instance will be
    created with *prop*. If *None*, use rc settings.

  *numpoints*: integer
    The number of points in the legend for line

  *scatterpoints*: integer
    The number of points in the legend for scatter plot

  *scatteroffsets*: list of floats
    a list of yoffsets for scatter symbols in legend

  *markerscale*: [ None | scalar ]
    The relative size of legend markers vs. original. If *None*, use rc
    settings.

  *fancybox*: [ None | False | True ]
    if True, draw a frame with a round fancybox.  If None, use rc
    
  *shadow*: [ None | False | True ]
    If *True*, draw a shadow behind legend. If *None*, use rc settings.

  *ncol* : integer
    number of columns. default is 1

  *mode* : [ "expand" | None ]
    if mode is "expand", the legend will be horizontally expanded
    to fill the axes area (or *bbox_to_anchor*)

  *bbox_to_anchor* : an instance of BboxBase or a tuple of 2 or 4 floats
    the bbox that the legend will be anchored.

  *bbox_transform* : [ an instance of Transform | None ]
    the transform for the bbox. transAxes if None.

  *title* : string
    the legend title

Padding and spacing between various elements use following keywords
parameters. The dimensions of these values are given as a fraction
of the fontsize. Values from rcParams will be used if None.

================   ==================================================================
Keyword            Description
================   ==================================================================
borderpad          the fractional whitespace inside the legend border
labelspacing       the vertical space between the legend entries
handlelength       the length of the legend handles
handletextpad      the pad between the legend handle and text
borderaxespad      the pad between the axes and legend border
columnspacing      the spacing between columns
================   ==================================================================


**Example:**

.. plot:: mpl_examples/api/legend_demo.py

Also see :ref:`plotting-guide-legend`.

"""
    ret =  gca().legend(*args, **kwargs)
    draw_if_interactive()
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def table(**kwargs):
    """call signature::

  table(cellText=None, cellColours=None,
        cellLoc='right', colWidths=None,
        rowLabels=None, rowColours=None, rowLoc='left',
        colLabels=None, colColours=None, colLoc='center',
        loc='bottom', bbox=None):

Add a table to the current axes.  Returns a
:class:`matplotlib.table.Table` instance.  For finer grained
control over tables, use the :class:`~matplotlib.table.Table`
class and add it to the axes with
:meth:`~matplotlib.axes.Axes.add_table`.

Thanks to John Gill for providing the class and table.

kwargs control the :class:`~matplotlib.table.Table`
properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  contains: a callable function
  figure: a :class:`matplotlib.figure.Figure` instance
  fontsize: a float in points
  gid: an id string
  label: any string
  lod: [True | False]
  picker: [None|float|boolean|callable]
  rasterized: [True | False | None]
  snap: unknown
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  visible: [True | False]
  zorder: any number         """
    ret =  gca().table(**kwargs)
    draw_if_interactive()
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def text(x, y, s, fontdict=None, withdash=False, **kwargs):
    """call signature::

  text(x, y, s, fontdict=None, **kwargs)

Add text in string *s* to axis at location *x*, *y*, data
coordinates.

Keyword arguments:

  *fontdict*:
    A dictionary to override the default text properties.
    If *fontdict* is *None*, the defaults are determined by your rc
    parameters.

  *withdash*: [ False | True ]
    Creates a :class:`~matplotlib.text.TextWithDash` instance
    instead of a :class:`~matplotlib.text.Text` instance.

Individual keyword arguments can be used to override any given
parameter::

    text(x, y, s, fontsize=12)

The default transform specifies that text is in data coords,
alternatively, you can specify text in axis coords (0,0 is
lower-left and 1,1 is upper-right).  The example below places
text in the center of the axes::

    text(0.5, 0.5,'matplotlib',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

You can put a rectangular box around the text instance (eg. to
set a background color) by using the keyword *bbox*.  *bbox* is
a dictionary of :class:`matplotlib.patches.Rectangle`
properties.  For example::

 text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))

Valid kwargs are :class:`matplotlib.text.Text` properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  backgroundcolor: any matplotlib color
  bbox: rectangle prop dict
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: any matplotlib color
  contains: a callable function
  family or fontfamily or fontname or name: [ FONTNAME | 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fontproperties or font_properties: a :class:`matplotlib.font_manager.FontProperties` instance
  gid: an id string
  horizontalalignment or ha: [ 'center' | 'right' | 'left' ]
  label: any string
  linespacing: float (multiple of font size)
  lod: [True | False]
  multialignment: ['left' | 'right' | 'center' ]
  picker: [None|float|boolean|callable]
  position: (x,y)
  rasterized: [True | False | None]
  rotation: [ angle in degrees | 'vertical' | 'horizontal' ]
  rotation_mode: unknown
  size or fontsize: [ size in points | 'xx-small' | 'x-small' | 'small' | 'medium' | 'large' | 'x-large' | 'xx-large' ]
  snap: unknown
  stretch or fontstretch: [ a numeric value in range 0-1000 | 'ultra-condensed' | 'extra-condensed' | 'condensed' | 'semi-condensed' | 'normal' | 'semi-expanded' | 'expanded' | 'extra-expanded' | 'ultra-expanded' ]
  style or fontstyle: [ 'normal' | 'italic' | 'oblique']
  text: string or anything printable with '%s' conversion.
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  variant or fontvariant: [ 'normal' | 'small-caps' ]
  verticalalignment or va or ma: [ 'center' | 'top' | 'bottom' | 'baseline' ]
  visible: [True | False]
  weight or fontweight: [ a numeric value in range 0-1000 | 'ultralight' | 'light' | 'normal' | 'regular' | 'book' | 'medium' | 'roman' | 'semibold' | 'demibold' | 'demi' | 'bold' | 'heavy' | 'extra bold' | 'black' ]
  x: float
  y: float
  zorder: any number         """
    ret =  gca().text(x, y, s, fontdict, withdash, **kwargs)
    draw_if_interactive()
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def annotate(*args, **kwargs):
    """call signature::

  annotate(s, xy, xytext=None, xycoords='data',
           textcoords='data', arrowprops=None, **kwargs)

Keyword arguments:

Annotate the *x*, *y* point *xy* with text *s* at *x*, *y*
location *xytext*.  (If *xytext* = *None*, defaults to *xy*,
and if *textcoords* = *None*, defaults to *xycoords*).

*arrowprops*, if not *None*, is a dictionary of line properties
(see :class:`matplotlib.lines.Line2D`) for the arrow that connects
annotation to the point.

If the dictionary has a key *arrowstyle*, a FancyArrowPatch
instance is created with the given dictionary and is
drawn. Otherwise, a YAArow patch instance is created and
drawn. Valid keys for YAArow are


=========   =============================================================
Key         Description
=========   =============================================================
width       the width of the arrow in points
frac        the fraction of the arrow length occupied by the head
headwidth   the width of the base of the arrow head in points
shrink      oftentimes it is convenient to have the arrowtip
            and base a bit away from the text and point being
            annotated.  If *d* is the distance between the text and
            annotated point, shrink will shorten the arrow so the tip
            and base are shink percent of the distance *d* away from the
            endpoints.  ie, ``shrink=0.05 is 5%``
?           any key for :class:`matplotlib.patches.polygon`
=========   =============================================================


Valid keys for FancyArrowPatch are


===============  ======================================================
Key              Description
===============  ======================================================
arrowstyle       the arrow style
connectionstyle  the connection style
relpos           default is (0.5, 0.5)
patchA           default is bounding box of the text
patchB           default is None
shrinkA          default is 2 points
shrinkB          default is 2 points
mutation_scale   default is text size (in points)
mutation_aspect  default is 1.
?                any key for :class:`matplotlib.patches.PathPatch`
===============  ======================================================


*xycoords* and *textcoords* are strings that indicate the
coordinates of *xy* and *xytext*.

=================   ===================================================
Property            Description
=================   ===================================================
'figure points'     points from the lower left corner of the figure
'figure pixels'     pixels from the lower left corner of the figure
'figure fraction'   0,0 is lower left of figure and 1,1 is upper, right
'axes points'       points from lower left corner of axes
'axes pixels'       pixels from lower left corner of axes
'axes fraction'     0,1 is lower left of axes and 1,1 is upper right
'data'              use the coordinate system of the object being
                    annotated (default)
'offset points'     Specify an offset (in points) from the *xy* value

'polar'             you can specify *theta*, *r* for the annotation,
                    even in cartesian plots.  Note that if you
                    are using a polar axes, you do not need
                    to specify polar for the coordinate
                    system since that is the native "data" coordinate
                    system.
=================   ===================================================

If a 'points' or 'pixels' option is specified, values will be
added to the bottom-left and if negative, values will be
subtracted from the top-right.  Eg::

  # 10 points to the right of the left border of the axes and
  # 5 points below the top border
  xy=(10,-5), xycoords='axes points'

Additional kwargs are Text properties:

  alpha: float (0.0 transparent through 1.0 opaque)
  animated: [True | False]
  axes: an :class:`~matplotlib.axes.Axes` instance
  backgroundcolor: any matplotlib color
  bbox: rectangle prop dict
  clip_box: a :class:`matplotlib.transforms.Bbox` instance
  clip_on: [True | False]
  clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]
  color: any matplotlib color
  contains: a callable function
  family or fontfamily or fontname or name: [ FONTNAME | 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
  figure: a :class:`matplotlib.figure.Figure` instance
  fontproperties or font_properties: a :class:`matplotlib.font_manager.FontProperties` instance
  gid: an id string
  horizontalalignment or ha: [ 'center' | 'right' | 'left' ]
  label: any string
  linespacing: float (multiple of font size)
  lod: [True | False]
  multialignment: ['left' | 'right' | 'center' ]
  picker: [None|float|boolean|callable]
  position: (x,y)
  rasterized: [True | False | None]
  rotation: [ angle in degrees | 'vertical' | 'horizontal' ]
  rotation_mode: unknown
  size or fontsize: [ size in points | 'xx-small' | 'x-small' | 'small' | 'medium' | 'large' | 'x-large' | 'xx-large' ]
  snap: unknown
  stretch or fontstretch: [ a numeric value in range 0-1000 | 'ultra-condensed' | 'extra-condensed' | 'condensed' | 'semi-condensed' | 'normal' | 'semi-expanded' | 'expanded' | 'extra-expanded' | 'ultra-expanded' ]
  style or fontstyle: [ 'normal' | 'italic' | 'oblique']
  text: string or anything printable with '%s' conversion.
  transform: :class:`~matplotlib.transforms.Transform` instance
  url: a url string
  variant or fontvariant: [ 'normal' | 'small-caps' ]
  verticalalignment or va or ma: [ 'center' | 'top' | 'bottom' | 'baseline' ]
  visible: [True | False]
  weight or fontweight: [ a numeric value in range 0-1000 | 'ultralight' | 'light' | 'normal' | 'regular' | 'book' | 'medium' | 'roman' | 'semibold' | 'demibold' | 'demi' | 'bold' | 'heavy' | 'extra bold' | 'black' ]
  x: float
  y: float
  zorder: any number

.. plot:: mpl_examples/pylab_examples/annotation_demo2.py"""
    ret =  gca().annotate(*args, **kwargs)
    draw_if_interactive()
    return ret

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def autumn():
    '''
    set the default colormap to autumn and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='autumn')
    im = gci()

    if im is not None:
        im.set_cmap(cm.autumn)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def bone():
    '''
    set the default colormap to bone and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='bone')
    im = gci()

    if im is not None:
        im.set_cmap(cm.bone)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cool():
    '''
    set the default colormap to cool and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='cool')
    im = gci()

    if im is not None:
        im.set_cmap(cm.cool)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def copper():
    '''
    set the default colormap to copper and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='copper')
    im = gci()

    if im is not None:
        im.set_cmap(cm.copper)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def flag():
    '''
    set the default colormap to flag and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='flag')
    im = gci()

    if im is not None:
        im.set_cmap(cm.flag)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def gray():
    '''
    set the default colormap to gray and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='gray')
    im = gci()

    if im is not None:
        im.set_cmap(cm.gray)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hot():
    '''
    set the default colormap to hot and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='hot')
    im = gci()

    if im is not None:
        im.set_cmap(cm.hot)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hsv():
    '''
    set the default colormap to hsv and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='hsv')
    im = gci()

    if im is not None:
        im.set_cmap(cm.hsv)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def jet():
    '''
    set the default colormap to jet and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='jet')
    im = gci()

    if im is not None:
        im.set_cmap(cm.jet)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pink():
    '''
    set the default colormap to pink and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='pink')
    im = gci()

    if im is not None:
        im.set_cmap(cm.pink)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def prism():
    '''
    set the default colormap to prism and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='prism')
    im = gci()

    if im is not None:
        im.set_cmap(cm.prism)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spring():
    '''
    set the default colormap to spring and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='spring')
    im = gci()

    if im is not None:
        im.set_cmap(cm.spring)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def summer():
    '''
    set the default colormap to summer and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='summer')
    im = gci()

    if im is not None:
        im.set_cmap(cm.summer)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def winter():
    '''
    set the default colormap to winter and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='winter')
    im = gci()

    if im is not None:
        im.set_cmap(cm.winter)
    draw_if_interactive()


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spectral():
    '''
    set the default colormap to spectral and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='spectral')
    im = gci()

    if im is not None:
        im.set_cmap(cm.spectral)
    draw_if_interactive()


