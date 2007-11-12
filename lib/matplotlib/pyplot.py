import sys

import matplotlib
from matplotlib import _pylab_helpers
from matplotlib.cbook import dedent, silent_list, is_string_like, is_numlike
from matplotlib.figure import Figure, figaspect
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.image import imread as _imread
from matplotlib import rcParams, rcParamsDefault, get_backend
from matplotlib.artist import getp, get
from matplotlib.artist import setp as _setp
from matplotlib.axes import Axes, PolarAxes
from matplotlib import mlab  # for csv2rec in plotfile

from matplotlib import cm
from matplotlib.cm import get_cmap

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



## Global ##

from matplotlib.backends import pylab_setup
new_figure_manager, draw_if_interactive, show = pylab_setup()

def switch_backend(newbackend):
    """
    Swtich the default backend to newbackend.  This feature is
    EXPERIMENTAL, and is only expected to work switching to an image
    backend.  Eg, if you have a bunch of PS scripts that you want to
    run from an interactive ipython session, you may want to switch to
    the PS backend before running them to avoid having a bunch of GUI
    windows popup.  If you try to interactively switch from one GUI
    backend to another, you will explode.

    Calling this command will close all open windows.
    """
    close('all')
    global new_figure_manager, draw_if_interactive, show
    matplotlib.use(newbackend)
    reload(backends)
    from backends import new_figure_manager, draw_if_interactive, show



def isinteractive():
    """
    Return the interactive status
    """
    return matplotlib.is_interactive()

def ioff():
    'turn interactive mode off'
    matplotlib.interactive(False)

def ion():
    'turn interactive mode on'
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
    get the current ScalarMappable instance (image or patch
    collection), or None if no images or patch collections have been
    defined.  The commands imshow and figimage create images
    instances, and the commands pcolor and scatter create patch
    collection instances
    """
    return gci._current
gci._current = None


def sci(im):
    """
    Set the current image (the target of colormap commands like jet, hot or clim)
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
    figure(num = None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


    Create a new figure and return a handle to it.  If num=None, the figure
    number will be incremented and a new figure will be created.  The returned
    figure objects have a .number attribute holding this number.

    If num is an integer, and figure(num) already exists, make it
    active and return the handle to it.  If figure(num) does not exist
    it will be created.  Numbering starts at 1, matlab style

      figure(1)

    If you are creating many figures, make sure you explicitly call "close"
    on the figures you are not using, because this will enable pylab
    to properly clean up the memory.

    kwargs:

      figsize - width x height in inches; defaults to rc figure.figsize
      dpi     - resolution; defaults to rc figure.dpi
      facecolor - the background color; defaults to rc figure.facecolor
      edgecolor - the border color; defaults to rc figure.edgecolor

    rcParams gives the default values from the matplotlibrc file

    FigureClass is a Figure or derived class that will be passed on to
    new_figure_manager in the backends which allows you to hook custom
    Figureclasses into the pylab interface.  Additional kwargs will be
    passed on to your figure init function
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

    figManager = _pylab_helpers.Gcf.get_fig_manager(num)
    if figManager is None:
        if get_backend()=='PS':  dpi = 72

        figManager = new_figure_manager(num, figsize=figsize,
                                             dpi=dpi,
                                             facecolor=facecolor,
                                             edgecolor=edgecolor,
                                             frameon=frameon,
                                             FigureClass=FigureClass,
                                             **kwargs)
        _pylab_helpers.Gcf.set_active(figManager)
        figManager.canvas.figure.number = num

    draw_if_interactive()
    return figManager.canvas.figure

def gcf():
    "Return a handle to the current figure"

    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is not None:
        return figManager.canvas.figure
    else:
        return figure()

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

    close() by itself closes the current figure

    close(num) closes figure number num

    close(h) where h is a figure handle(instance) closes that figure

    close('all') closes all the figure windows
    """

    if len(args)==0:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is None: return
        else: _pylab_helpers.Gcf.destroy(figManager.num)
    elif len(args)==1:
        arg = args[0]
        if arg=='all':
            for manager in _pylab_helpers.Gcf.get_all_fig_managers():
                _pylab_helpers.Gcf.destroy(manager.num)
        elif isinstance(arg, int):
            _pylab_helpers.Gcf.destroy(arg)
        elif isinstance(arg, Figure):
            for manager in _pylab_helpers.Gcf.get_all_fig_managers():
                if manager.canvas.figure==arg:
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

# Putting things in figures

def figtext(*args, **kwargs):

    ret =  gcf().text(*args, **kwargs)
    draw_if_interactive()
    return ret
if Figure.text.__doc__ is not None:
    figtext.__doc__ = dedent(Figure.text.__doc__)

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
    Place a legend in the figure.  Labels are a sequence of
    strings, handles is a sequence of line or patch instances, and
    loc can be a string r an integer specifying the legend
    location

    USAGE:
      legend( (line1, line2, line3),
              ('label1', 'label2', 'label3'),
              'upper right')

    See help(legend) for information about the location codes

    A matplotlib.legend.Legend instance is returned
    """
    l=  gcf().legend(handles, labels, loc, **kwargs)
    draw_if_interactive()
    return l



## Figure and Axes hybrid ##

def hold(b=None):
    """
    Set the hold state.  If hold is None (default), toggle the
    hold state.  Else set the hold state to boolean value b.

    Eg
    hold()      # toggle hold
    hold(True)  # hold is on
    hold(False) # hold is off

    When hold is True, subsequent plot commands will be added to the
    current axes.  When hold is False, the current axes and figure
    will be cleared on the next plot command
    """

    gcf().hold(b)
    gca().hold(b)
    rc('axes', hold=b)

def ishold():
    """
    Return the hold status of the current axes
    """
    return gca().ishold()

def over(func, *args, **kwargs):
    """
    Call func(*args, **kwargs) with hold(True) and then restore the hold state
    """
    h = ishold()
    hold(True)
    func(*args, **kwargs)
    hold(h)



## Axes ##

def axes(*args, **kwargs):
    """
    Add an axes at positon rect specified by::

    axes() by itself creates a default full subplot(111) window axis

    axes(rect, axisbg='w') where rect=[left, bottom, width, height] in
    normalized (0,1) units.  axisbg is the background color for the
    axis, default white

    axes(h) where h is an axes instance makes h the
    current axis An Axes instance is returned

    kwargs:

      axisbg=color   : the axes background color
      frameon=False  : don't display the frame
      sharex=otherax : the current axes shares xaxis attribute with otherax
      sharey=otherax : the current axes shares yaxis attribute with otherax
      polar=True|False : use a polar axes or not

    Examples

      examples/axes_demo.py places custom axes.
      examples/shared_axis_demo.py uses sharex and sharey


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
    delaxes(ax) - remove ax from the current figure.  If ax doesn't
    exist an error will be raised.

    delaxes(): delete the current axes
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
    axis properties either using set or the Axes methods.

    Example:

      plot(t,s)
      set(gca(), 'xlim', [0,10])  # set the x axis limits

    or

      plot(t,s)
      a = gca()
      a.set_xlim([0,10])          # does the same
    """

    ax =  gcf().gca(**kwargs)
    return ax

# More ways of creating axes:

def subplot(*args, **kwargs):
    """
    Create a subplot command, creating axes with

      subplot(numRows, numCols, plotNum)

    where plotNum=1 is the first plot number and increasing plotNums
    fill rows first.  max(plotNum)==numRows*numCols

    You can leave out the commas if numRows<=numCols<=plotNum<10, as
    in

      subplot(211)    # 2 rows, 1 column, first (upper) plot

    subplot(111) is the default axis

    The background color of the subplot can be specified via keyword
    argument 'axisbg', which takes a color string or gdk.Color as value, as in

    subplot(211, axisbg='y')

    See help(axes) for additional information on axes and subplot
    keyword arguments.

    New subplots that overlap old will delete the old axes.  If you do
    not want this behavior, use fig.add_subplot or the axes command.  Eg

      from pylab import *
      plot([1,2,3])  # implicitly creates subplot(111)
      subplot(211)   # overlaps, subplot(111) is killed
      plot(rand(12), rand(12))

    """


    fig = gcf()
    a = fig.add_subplot(*args, **kwargs)
    bbox = a.bbox
    byebye = []
    for other in fig.axes:
        if other==a: continue
        if bbox.overlaps(other.bbox, ignoreend=True):
            byebye.append(other)
    for ax in byebye: delaxes(ax)

    draw_if_interactive()
    return a


def twinx(ax=None):
    """
    Make a second axes overlay ax (or the current axes if ax is None)
    sharing the xaxis.  The ticks for ax2 will be placed on the right,
    and the ax2 instance is returned.  See examples/two_scales.py
    """
    if ax is None:
        ax=gca()
    draw_if_interactive()
    return ax.twinx()


def twiny(ax=None):
    """
    Make a second axes overlay ax (or the current axes if ax is None)
    sharing the yaxis.  The ticks for ax2 will be placed on the top,
    and the ax2 instance is returned.
    """
    if ax is None:
        ax=gca()
    draw_if_interactive()
    return ax.twiny()



def subplots_adjust(*args, **kwargs):
    """
    subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=None, hspace=None)

    Tune the subplot layout via the figure.SubplotParams mechanism.
    The parameter meanings (and suggested defaults) are

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
    Launch a subplot tool window for targetfig (default gcf)

    A matplotlib.widgets.SubplotTool instance is returned
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
    Turn the axes box on or off according to 'on'

    If on is None, toggle state
    """
    ax = gca()
    if on is None:
        on = not ax.get_frame_on()
    ax.set_frame_on(on)
    draw_if_interactive()

def title(s, *args, **kwargs):
    """
    Set the title of the current axis to s

    Default font override is:
      override = {
        'fontsize'            : 'medium',
        'verticalalignment'   : 'bottom',
        'horizontalalignment' : 'center'
      }

    See the text docstring for information of how override and the
    optional args work

    """
    l =  gca().set_title(s, *args, **kwargs)
    draw_if_interactive()
    return l




## Axis ##

def axis(*v, **kwargs):
    """
    Set/Get the axis properties::

        v = axis()  returns the current axes as v = [xmin, xmax, ymin, ymax]

        axis(v) where v = [xmin, xmax, ymin, ymax] sets the min and max
          of the x and y axes

        axis('off') turns off the axis lines and labels

        axis('equal') changes limits of x or y axis so that equal
          increments of x and y have the same length; a circle
          is circular.

        axis('scaled') achieves the same result by changing the
          dimensions of the plot box instead of the axis data
          limits.

        axis('tight') changes x and y axis limits such that all data is
          shown. If all data is already shown, it will move it to the center
          of the figure without modifying (xmax-xmin) or (ymax-ymin). Note
          this is slightly different than in matlab.

        axis('image') is 'scaled' with the axis limits equal to the
          data limits.

        axis('auto') or 'normal' (deprecated) restores default behavior;
          axis limits are automatically scaled to make the data fit
          comfortably within the plot box.

       if len(*v)==0, you can pass in xmin, xmax, ymin, ymax as kwargs
       selectively to alter just those limits w/o changing the others.
       See help(xlim) and help(ylim) for more information

       The xmin, xmax, ymin, ymax tuple is returned

    """
    ax = gca()
    v = ax.axis(*v, **kwargs)
    draw_if_interactive()
    return v

def xlabel(s, *args, **kwargs):
    """
    Set the x axis label of the current axis to s

    Default override is

      override = {
          'fontsize'            : 'small',
          'verticalalignment'   : 'top',
          'horizontalalignment' : 'center'
          }

    See the text docstring for information of how override and
    the optional args work

    """
    l =  gca().set_xlabel(s, *args, **kwargs)
    draw_if_interactive()
    return l

def ylabel(s, *args, **kwargs):
    """
    Set the y axis label of the current axis to s

    Defaults override is

        override = {
           'fontsize'            : 'small',
           'verticalalignment'   : 'center',
           'horizontalalignment' : 'right',
           'rotation'='vertical' : }

    See the text docstring for information of how override and the
    optional args work

    """
    l = gca().set_ylabel(s, *args, **kwargs)
    draw_if_interactive()
    return l





def xlim(*args, **kwargs):
    """
    Set/Get the xlimits of the current axes

    xmin, xmax = xlim()   : return the current xlim
    xlim( (xmin, xmax) )  : set the xlim to xmin, xmax
    xlim( xmin, xmax )    : set the xlim to xmin, xmax

    If you do not specify args, you can pass the xmin and xmax as
    kwargs, eg

      xlim(xmax=3) # adjust the max leaving min unchanged
      xlim(xmin=1) # adjust the min leaving max unchanged

    The new axis limits are returned as a length 2 tuple

    """
    ax = gca()
    ret = ax.set_xlim(*args, **kwargs)
    draw_if_interactive()
    return ret


def ylim(*args, **kwargs):
    """
    Set/Get the ylimits of the current axes

    ymin, ymax = ylim()   : return the current ylim
    ylim( (ymin, ymax) )  : set the ylim to ymin, ymax
    ylim( ymin, ymax )    : set the ylim to ymin, ymax

    If you do not specify args, you can pass the ymin and ymax as
    kwargs, eg

      ylim(ymax=3) # adjust the max leaving min unchanged
      ylim(ymin=1) # adjust the min leaving max unchanged

    The new axis limits are returned as a length 2 tuple

    """
    ax = gca()
    ret = ax.set_ylim(*args, **kwargs)
    draw_if_interactive()
    return ret




def xticks(*args, **kwargs):
    """
    Set/Get the xlimits of the current ticklocs, labels

    # return locs, labels where locs is an array of tick locations and
    # labels is an array of tick labels.
    locs, labels = xticks()

    # set the locations of the xticks
    xticks( arange(6) )

    # set the locations and labels of the xticks
    xticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )

    The keyword args, if any, are text properties; see text for more
    information on text properties.
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
    Set/Get the ylimits of the current ticklocs, labels

    # return locs, labels where locs is an array of tick locations and
    # labels is an array of tick labels.
    locs, labels = yticks()

    # set the locations of the yticks
    yticks( arange(6) )

    # set the locations and labels of the yticks
    yticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )

    The keyword args, if any, are text properties; see text for more
    information on text properties.
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



def rgrids(*args, **kwargs):
    """
    Set/Get the radial locations of the gridlines and ticklabels

    With no args, simply return lines, labels where lines is an
    array of radial gridlines (Line2D instances) and labels is an
    array of tick labels (Text instances).

      lines, labels = rgrids()

    With arguments, the syntax is

      lines, labels = RGRIDS(radii, labels=None, angle=22.5, **kwargs)

    The labels will appear at radial distances radii at angle

      labels, if not None, is a len(radii) list of strings of the
      labels to use at each angle.

      if labels is None, the self.rformatter will be used

    Return value is a list of lines, labels where the lines are
    matplotlib.Line2D instances and the labels are matplotlib.Text
    instances.  Note that on input the labels argument is a list of
    strings, and on output it is a list of Text instances

    Examples
      # set the locations of the radial gridlines and labels
      lines, labels = rgrids( (0.25, 0.5, 1.0) )

      # set the locations and labels of the radial gridlines and labels
      lines, labels = rgrids( (0.25, 0.5, 1.0), ('Tom', 'Dick', 'Harry' )

    """
    ax = gca()
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('rgrids only defined for polar axes')
    if len(args)==0:
        lines = ax.rgridlines()
        labels = ax.rgridlabels()
    else:
        lines, labels = ax.set_rgrids(*args, **kwargs)

    draw_if_interactive()
    return ( silent_list('Line2D rgridline', lines),
             silent_list('Text rgridlabel', labels) )

def thetagrids(*args, **kwargs):
    """
    Set/Get the theta locations of the gridlines and ticklabels

    If no arguments are passed, return lines, labels where lines is an
    array of radial gridlines (Line2D instances) and labels is an
    array of tick labels (Text instances).

      lines, labels = thetagrids()

    Otherwise the syntax is

      lines, labels = THETAGRIDS(angles, labels=None, fmt='%d', frac = 1.1)

    set the angles at which to place the theta grids (these gridlines
    are equal along the theta dimension).  angles is in degrees

      labels, if not None, is a len(angles) list of strings of the
      labels to use at each angle.

      if labels is None, the labels with be fmt%angle

      frac is the fraction of the polar axes radius at which to place
      the label (1 is the edge).Eg 1.05 isd outside the axes and 0.95
      is inside the axes

    Return value is a list of lines, labels where the lines are
    matplotlib.Line2D instances and the labels are matplotlib.Text
    instances.  Note that on input the labels argument is a list of
    strings, and on output it is a list of Text instances

    Examples:

      # set the locations of the radial gridlines and labels
      lines, labels = thetagrids( range(45,360,90) )

      # set the locations and labels of the radial gridlines and labels
      lines, labels = thetagrids( range(45,360,90), ('NE', 'NW', 'SW','SE') )
    """
    ax = gca()
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('rgrids only defined for polar axes')
    if len(args)==0:
        lines = ax.thetagridlines()
        labels = ax.thetagridlabels()
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
    axes     - Create a new axes
    axis     - Set or return the current axis limits
    bar      - make a bar chart
    boxplot  - make a box and whiskers chart
    cla      - clear current axes
    clabel   - label a contour plot
    clf      - clear a figure window
    close    - close a figure window
    colorbar - add a colorbar to the current figure
    cohere   - make a plot of coherence
    contour  - make a contour plot
    contourf  - make a filled contour plot
    csd      - make a plot of cross spectral density
    draw     - force a redraw of the current figure
    errorbar - make an errorbar graph
    figlegend - add a legend to the figure
    figimage - add an image to the figure, w/o resampling
    figtext  - add text in figure coords
    figure   - create or change active figure
    fill     - make filled polygons
    gca      - return the current axes
    gcf      - return the current figure
    gci      - get the current image, or None
    get      - get a handle graphics property
    hist     - make a histogram
    hold     - set the hold state on current axes
    legend   - add a legend to the axes
    loglog   - a log log plot
    imread   - load image file into array
    imshow   - plot image data
    matshow  - display a matrix in a new figure preserving aspect
    pcolor   - make a pseudocolor plot
    plot     - make a line plot
    plotfile - plot data from a flat file
    psd      - make a plot of power spectral density
    quiver   - make a direction field (arrows) plot
    rc       - control the default params
    savefig  - save the current figure
    scatter  - make a scatter plot
    set      - set a handle graphics property
    semilogx - log x axis
    semilogy - log y axis
    show     - show the figures
    specgram - a spectrogram plot
    stem     - make a stem plot
    subplot  - make a subplot (numrows, numcols, axesnum)
    table    - add a table to the axes
    text     - add some text at location x,y to the current axes
    title    - add a title to the current axes
    xlabel   - add an xlabel to the current axes
    ylabel   - add a ylabel to the current axes

    autumn - set the default colormap to autumn
    bone   - set the default colormap to bone
    cool   - set the default colormap to cool
    copper - set the default colormap to copper
    flag   - set the default colormap to flag
    gray   - set the default colormap to gray
    hot    - set the default colormap to hot
    hsv    - set the default colormap to hsv
    jet    - set the default colormap to jet
    pink   - set the default colormap to pink
    prism  - set the default colormap to prism
    spring - set the default colormap to spring
    summer - set the default colormap to summer
    winter - set the default colormap to winter
    spectral - set the default colormap to spectral

    """
    pass


def get_plot_commands(): return ( 'axes', 'axis', 'bar', 'boxplot', 'cla', 'clf',
    'close', 'colorbar', 'cohere', 'csd', 'draw', 'errorbar',
    'figlegend', 'figtext', 'figimage', 'figure', 'fill', 'gca',
    'gcf', 'gci', 'get', 'gray', 'barh', 'jet', 'hist', 'hold', 'imread',
    'imshow', 'legend', 'loglog', 'quiver', 'rc', 'pcolor', 'pcolormesh', 'plot', 'psd',
    'savefig', 'scatter', 'set', 'semilogx', 'semilogy', 'show',
    'specgram', 'stem', 'subplot', 'table', 'text', 'title', 'xlabel',
    'ylabel', 'pie', 'polar')

def colors():
    """
    This is a do nothing function to provide you with help on how
    matplotlib handles colors.

    Commands which take color arguments can use several formats to
    specify the colors.  For the basic builtin colors, you can use a
    single letter

      b  : blue
      g  : green
      r  : red
      c  : cyan
      m  : magenta
      y  : yellow
      k  : black
      w  : white


    For a greater range of colors, you have two options.  You can
    specify the color using an html hex string, as in

      color = '#eeefff'

    or you can pass an R,G,B tuple, where each of R,G,B are in the
    range [0,1].

    You can also use any legal html name for a color, like 'red',
    'burlywood' and 'chartreuse'

    The example below creates a subplot with a dark
    slate gray background

       subplot(111, axisbg=(0.1843, 0.3098, 0.3098))

    Here is an example that creates a pale turqoise title

      title('Is this the best color?', color='#afeeee')

    """
    pass



def colormaps():
    """
    matplotlib provides the following colormaps.

      autumn bone cool copper flag gray hot hsv jet pink prism
      spring summer winter spectral

    You can set the colormap for an image, pcolor, scatter, etc,
    either as a keyword argumentdef con

    >>> imshow(X, cmap=cm.hot)

    or post-hoc using the corresponding pylab interface function

    >>> imshow(X)
    >>> hot()
    >>> jet()

    In interactive mode, this will update the colormap allowing you to
    see which one works best for your data.
    """
    pass


## Plotting part 1: manually generated functions and wrappers ##


from matplotlib.colorbar import colorbar_doc
def colorbar(mappable=None, cax=None, ax=None, **kw):
    if mappable is None:
        mappable = gci()
    ret = gcf().colorbar(mappable, cax = cax, ax=ax, **kw)
    draw_if_interactive()
    return ret
colorbar.__doc__ = colorbar_doc

def clim(vmin=None, vmax=None):
    """
    Set the color limits of the current image

    To apply clim to all axes images do

    clim(0, 0.5)

    If either vmin or vmax is None, the image min/max respectively
    will be used for color scaling.

    If you want to set the clim of multiple images,
    use, for example for im in gca().get_images(): im.set_clim(0,
    0.05)

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

def matshow(A, fignum=None, **kw):
    """Display an array as a matrix in a new figure window.

    The origin is set at the upper left hand corner and rows (first dimension
    of the array) are displayed horizontally.  The aspect ratio of the figure
    window is that of the array, unless this would make an excessively
    short or narrow figure.

    Tick labels for the xaxis are placed on top.

    With one exception, keyword arguments are passed to
    imshow().

    Special keyword argument which is NOT passed to imshow():

      - fignum(None): by default, matshow() creates a new figure window with
      automatic numbering.  If fignum is given as an integer, the created
      figure will use this figure number.  Because of how matshow() tries to
      set the figure aspect ratio to be the one of the array, if you provide
      the number of an already existing figure, strange things may happen.

      if fignum is False or 0, a new figure window will NOT be created.

    Example usage:

    def samplemat(dims):
        aa = zeros(dims)
        for i in range(min(dims)):
            aa[i,i] = i
        return aa

    dimlist = [(12,12),(128,64),(64,512),(2048,256)]

    for d in dimlist:
        im = matshow(samplemat(d))
    show()
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
    POLAR(theta, r)

    Make a polar plot.  Multiple theta, r arguments are supported,
    with format strings, as in plot.
    """
    ax = gca(polar=True)
    ret = ax.plot(*args, **kwargs)
    draw_if_interactive()
    return ret

def plotfile(fname, cols=(0,), plotfuncs=None,
             comments='#', skiprows=0, checkrows=5, delimiter=',',
             **kwargs):
    """
    plot the data in fname

    cols is a sequence of column identifiers to plot.  An identifier
    is either an int or a string.  if it is an int, it indicates the
    column number.  If it is a string, it indicates the column header.
    mpl will make column headers lower case, replace spaces with
    underscores, and remove all illegal characters; so 'Adj Close*'
    will have name 'adj_close'

    if len(cols)==1, only that column will be plotted on the y axis.
    if len(cols)>1, the first element will be an identifier for data
    for the x axis and the remaining elements will be the column
    indexes for multiple subplots

    plotfuncs, if not None, is a dictionary mapping identifier to an
    Axes plotting function as a string.  Default is 'plot', other
    choices are 'semilogy', 'fill', 'bar', etc...  You must use the
    same type of identifier in the cols vector as you use in the
    plotfuncs dictionary, eg integer column numbers in both or column
    names in both.

    comments, skiprows, checkrows, and delimiter are all passed on to
    matplotlib.mlab.csv2rec to load the data into a record array.  See
    the help there fore more information.

    kwargs are passed on to plotting functions

    Example usage:

      # plot the 2nd and 4th column against the 1st in two subplots
      plotfile(fname, (0,1,3))

      # plot using column names; specify an alternate plot type for volume
      plotfile(fname, ('date', 'volume', 'adj_close'), plotfuncs={'volume': 'semilogy'})
    """

    fig = figure()
    if len(cols)<1:
        raise ValueError('must have at least one column of data')

    if plotfuncs is None:
        plotfuncs = dict()
    r = mlab.csv2rec(fname, comments=comments,
                skiprows=skiprows, checkrows=checkrows, delimiter=delimiter)

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

    if len(cols)==1:
        ax1 = fig.add_subplot(N,1,i)
        funcname = plotfuncs.get(cols[0], 'plot')
        func = getattr(ax1, funcname)
        func(x, **kwargs)
        ax1.set_xlabel(xname)
    else:
        N = len(cols)
        for i in range(1,N):
            if i==1:
                ax = ax1 = fig.add_subplot(N-1,1,i)
                ax.grid(True)
            else:
                ax = fig.add_subplot(N-1,1,i, sharex=ax1)
                ax.grid(True)


            yname, y = getname_val(cols[i])

            funcname = plotfuncs.get(cols[i], 'plot')
            func = getattr(ax, funcname)

            func(x, y, **kwargs)
            ax.set_ylabel(yname)
            if ax.is_last_row():
                ax.set_xlabel(xname)
            else:
                ax.set_xlabel('')


    if xname=='date':
        fig.autofmt_xdate()

    draw_if_interactive()


## Plotting part 2: autogenerated wrappers for axes methods ##


### Do not edit below this point
# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def acorr(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().acorr(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.acorr.__doc__ is not None:
    acorr.__doc__ = dedent(Axes.acorr.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def arrow(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().arrow(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.arrow.__doc__ is not None:
    arrow.__doc__ = dedent(Axes.arrow.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axhline(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axhline(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.axhline.__doc__ is not None:
    axhline.__doc__ = dedent(Axes.axhline.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axhspan(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axhspan(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.axhspan.__doc__ is not None:
    axhspan.__doc__ = dedent(Axes.axhspan.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axvline(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axvline(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.axvline.__doc__ is not None:
    axvline.__doc__ = dedent(Axes.axvline.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axvspan(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axvspan(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.axvspan.__doc__ is not None:
    axvspan.__doc__ = dedent(Axes.axvspan.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def bar(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().bar(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.bar.__doc__ is not None:
    bar.__doc__ = dedent(Axes.bar.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def barh(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().barh(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.barh.__doc__ is not None:
    barh.__doc__ = dedent(Axes.barh.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def broken_barh(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().broken_barh(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.broken_barh.__doc__ is not None:
    broken_barh.__doc__ = dedent(Axes.broken_barh.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def boxplot(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().boxplot(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.boxplot.__doc__ is not None:
    boxplot.__doc__ = dedent(Axes.boxplot.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cohere(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().cohere(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.cohere.__doc__ is not None:
    cohere.__doc__ = dedent(Axes.cohere.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def clabel(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().clabel(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.clabel.__doc__ is not None:
    clabel.__doc__ = dedent(Axes.clabel.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def contour(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().contour(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    if ret._A is not None: gci._current = ret
    hold(b)
    return ret
if Axes.contour.__doc__ is not None:
    contour.__doc__ = dedent(Axes.contour.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def contourf(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().contourf(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    if ret._A is not None: gci._current = ret
    hold(b)
    return ret
if Axes.contourf.__doc__ is not None:
    contourf.__doc__ = dedent(Axes.contourf.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def csd(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().csd(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.csd.__doc__ is not None:
    csd.__doc__ = dedent(Axes.csd.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def errorbar(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().errorbar(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.errorbar.__doc__ is not None:
    errorbar.__doc__ = dedent(Axes.errorbar.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def fill(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().fill(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.fill.__doc__ is not None:
    fill.__doc__ = dedent(Axes.fill.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hist(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().hist(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.hist.__doc__ is not None:
    hist.__doc__ = dedent(Axes.hist.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hlines(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().hlines(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.hlines.__doc__ is not None:
    hlines.__doc__ = dedent(Axes.hlines.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def imshow(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().imshow(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret
    hold(b)
    return ret
if Axes.imshow.__doc__ is not None:
    imshow.__doc__ = dedent(Axes.imshow.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def loglog(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().loglog(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.loglog.__doc__ is not None:
    loglog.__doc__ = dedent(Axes.loglog.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pcolor(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().pcolor(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret
    hold(b)
    return ret
if Axes.pcolor.__doc__ is not None:
    pcolor.__doc__ = dedent(Axes.pcolor.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pcolormesh(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().pcolormesh(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret
    hold(b)
    return ret
if Axes.pcolormesh.__doc__ is not None:
    pcolormesh.__doc__ = dedent(Axes.pcolormesh.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pie(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().pie(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.pie.__doc__ is not None:
    pie.__doc__ = dedent(Axes.pie.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def plot(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().plot(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.plot.__doc__ is not None:
    plot.__doc__ = dedent(Axes.plot.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def plot_date(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().plot_date(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.plot_date.__doc__ is not None:
    plot_date.__doc__ = dedent(Axes.plot_date.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def psd(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().psd(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.psd.__doc__ is not None:
    psd.__doc__ = dedent(Axes.psd.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def quiver(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().quiver(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret
    hold(b)
    return ret
if Axes.quiver.__doc__ is not None:
    quiver.__doc__ = dedent(Axes.quiver.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def quiverkey(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().quiverkey(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.quiverkey.__doc__ is not None:
    quiverkey.__doc__ = dedent(Axes.quiverkey.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def scatter(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().scatter(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret
    hold(b)
    return ret
if Axes.scatter.__doc__ is not None:
    scatter.__doc__ = dedent(Axes.scatter.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def semilogx(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().semilogx(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.semilogx.__doc__ is not None:
    semilogx.__doc__ = dedent(Axes.semilogx.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def semilogy(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().semilogy(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.semilogy.__doc__ is not None:
    semilogy.__doc__ = dedent(Axes.semilogy.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def specgram(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().specgram(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret[-1]
    hold(b)
    return ret
if Axes.specgram.__doc__ is not None:
    specgram.__doc__ = dedent(Axes.specgram.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spy(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().spy(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise
    gci._current = ret
    hold(b)
    return ret
if Axes.spy.__doc__ is not None:
    spy.__doc__ = dedent(Axes.spy.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def stem(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().stem(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.stem.__doc__ is not None:
    stem.__doc__ = dedent(Axes.stem.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def step(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().step(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.step.__doc__ is not None:
    step.__doc__ = dedent(Axes.step.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def vlines(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().vlines(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.vlines.__doc__ is not None:
    vlines.__doc__ = dedent(Axes.vlines.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def xcorr(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = kwargs.pop('hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().xcorr(*args, **kwargs)
        draw_if_interactive()
    except:
        hold(b)
        raise

    hold(b)
    return ret
if Axes.xcorr.__doc__ is not None:
    xcorr.__doc__ = dedent(Axes.xcorr.__doc__) + """
Additional kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cla(*args, **kwargs):

    ret =  gca().cla(*args, **kwargs)
    draw_if_interactive()
    return ret
if Axes.cla.__doc__ is not None:
    cla.__doc__ = dedent(Axes.cla.__doc__)

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def grid(*args, **kwargs):

    ret =  gca().grid(*args, **kwargs)
    draw_if_interactive()
    return ret
if Axes.grid.__doc__ is not None:
    grid.__doc__ = dedent(Axes.grid.__doc__)

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def legend(*args, **kwargs):

    ret =  gca().legend(*args, **kwargs)
    draw_if_interactive()
    return ret
if Axes.legend.__doc__ is not None:
    legend.__doc__ = dedent(Axes.legend.__doc__)

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def table(*args, **kwargs):

    ret =  gca().table(*args, **kwargs)
    draw_if_interactive()
    return ret
if Axes.table.__doc__ is not None:
    table.__doc__ = dedent(Axes.table.__doc__)

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def text(*args, **kwargs):

    ret =  gca().text(*args, **kwargs)
    draw_if_interactive()
    return ret
if Axes.text.__doc__ is not None:
    text.__doc__ = dedent(Axes.text.__doc__)

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def annotate(*args, **kwargs):

    ret =  gca().annotate(*args, **kwargs)
    draw_if_interactive()
    return ret
if Axes.annotate.__doc__ is not None:
    annotate.__doc__ = dedent(Axes.annotate.__doc__)

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




