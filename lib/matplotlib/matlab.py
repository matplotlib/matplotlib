"""
This is a matlab style functional interface the matplotlib.

The following plotting commands are provided; some of these do not
exist in matlab but have proven themselves to be useful nonetheless.
The majority of them, however, are matlab commands

_Plotting commands

  axes     - Create a new axes
  axhline  - draw a horizontal line across axes
  axvline  - draw a vertical line across axes
  axhspan  - draw a horizontal bar across axes
  axvspan  - draw a vertical bar across axes 
  axis     - Set or return the current axis limits
  bar      - make a bar chart
  barh     - a horizontal bar chart  
  cla      - clear current axes
  clf      - clear a figure window
  close    - close a figure window
  colorbar - add a colorbar to the current figure
  cohere   - make a plot of coherence
  csd      - make a plot of cross spectral density  
  draw     - Force a redraw of the current figure
  errorbar - make an errorbar graph
  figlegend - make legend on the figure rather than the axes
  figimage  - make a figure image
  figtext   - add text in figure coords
  figure   - create or change active figure
  fill     - make filled polygons
  gca      - return the current axes
  gcf      - return the current figure
  gci      - get the current image, or None
  get      - get a handle graphics property
  gray     - set the current colormap to gray
  jet      - set the current colormap to jet
  hist     - make a histogram
  hold     - set the axes hold state
  legend   - make an axes legend
  loglog   - a log log plot
  imread   - load image file into array
  imshow   - plot image data
  pcolor   - make a pseudocolor plot
  plot     - make a line plot
  psd      - make a plot of power spectral density
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
  table    - add a table to the plot
  text     - add some text at location x,y to the current axes
  title    - add a title to the current axes
  xlim     - set/get the xlimits
  ylim     - set/get the ylimits
  xticks   - set/get the xticks
  yticks   - set/get the yticks
  xlabel   - add an xlabel to the current axes
  ylabel   - add a ylabel to the current axes
  
_Matrix commands

  cumprod   - the cumulative product along a dimension
  cumsum    - the cumulative sum along a dimension
  detrend   - remove the mean or besdt fit line from an array
  diag      - the k-th diagonal of matrix 
  diff      - the n-th differnce of an array
  eig       - the eigenvalues and eigen vectors of v
  eye       - a matrix where the k-th diagonal is ones, else zero 
  find      - return the indices where a condition is nonzero  
  fliplr    - flip the rows of a matrix up/down
  flipud    - flip the columns of a matrix left/right
  linspace  - a linear spaced vector of N values from min to max inclusive
  meshgrid  - repeat x and y to make regular matrices
  ones      - an array of ones
  rand      - an array from the uniform distribution [0,1]
  randn     - an array from the normal distribution
  rot90     - rotate matrix k*90 degress counterclockwise
  squeeze   - squeeze an array removing any dimensions of length 1
  tri       - a triangular matrix
  tril      - a lower triangular matrix
  triu      - an upper triangular matrix
  vander    - the Vandermonde matrix of vector x
  svd       - singular value decomposition
  zeros     - a matrix of zeros
  
_Probability

  levypdf   - The levy probability density function from the char. func.
  normpdf   - The Gaussian probability density function
  rand      - random numbers from the uniform distribution
  randn     - random numbers from the normal distribution

_Statistics

  corrcoef  - correlation coefficient
  cov       - covariance matrix
  max       - the maximum along dimension m
  mean      - the mean along dimension m
  median    - the median along dimension m
  min       - the minimum along dimension m
  norm      - the norm of vector x
  prod      - the product along dimension m
  ptp       - the max-min along dimension m
  std       - the standard deviation along dimension m
  sum       - the sum along dimension m

_Time series analysis

  bartlett  - M-point Bartlett window
  blackman  - M-point Blackman window
  cohere    - the coherence using average periodiogram
  csd       - the cross spectral density using average periodiogram
  fft       - the fast Fourier transform of vector x
  hamming   - M-point Hamming window
  hanning   - M-point Hanning window
  hist      - compute the histogram of x
  kaiser    - M length Kaiser window
  psd       - the power spectral density using average periodiogram
  sinc      - the sinc function of array x

_Dates

  date2num  - convert python datetimes to numeric representation
  drange    - create an array of numbers for date plots
  num2date  - convert numeric type (float days since 0001) to datetime
  
_Other

  angle     - the angle of a complex array
  load     - load ASCII data into array
  polyfit   - fit x, y to an n-th order polynomial
  polyval   - evaluate an n-th order polynomial
  roots     - the roots of the polynomial coefficients in p
  save      - save an array to an ASCII file
  trapz     - trapezoidal integration

__end

Credits: The plotting commands were provided by
John D. Hunter <jdhunter@ace.bsd.uhicago.edu>

Most of the other commands are from the Numeric, MLab and FFT, with
the exception of those in mlab.py provided by matplotlib.
"""

# bring all the MLab and mlab symbols in so folks can import them from
# matplotlib.matlab in one fell swoop
from numerix import *
from mlab import *

import cm
import _matlab_helpers
import mlab  #so I can override hist, psd, etc...

from axes import Axes
from backends import new_figure_manager, error_msg, \
     draw_if_interactive, show
from cbook import flatten, is_string_like, exception_to_str
from colors import normalize
from cm import ColormapJet, Grayscale, get_cmap
from figure import Figure
import image
from matplotlib import rcParams, rcParamsDefault, get_backend
from backend_bases import FigureCanvasBase
from image import imread as _imread
from patches import Polygon
from transforms import blend_xy_sep_transform

# catch more than an import error here, since the src could fail too,
# eg a bad pytz install.  I don't want to break all of matplotlib for
# date support
try: from dates import *
except: pass 

import matplotlib

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
    range [0,1].  The example below creates a subplot with a dark
    slate gray background

       subplot(111, axisbg=(0.1843, 0.3098, 0.3098))

    Here is an example that creates a pale turqoise title

      title('Is this the best color?', color='#afeeee')

    """
    pass

def plotting():
    """
    Plotting commands
    axes     - Create a new axes
    axis     - Set or return the current axis limits
    bar      - make a bar chart
    cla      - clear current axes
    clf      - clear a figure window
    close    - close a figure window
    colorbar - add a colorbar to the current figure
    cohere   - make a plot of coherence
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
    gray     - set the current colormap to gray
    jet      - set the current colormap to jet
    hist     - make a histogram
    hold     - set the hold state on current axes
    legend   - add a legend to the axes
    loglog   - a log log plot
    imread   - load image file into array
    imshow   - plot image data
    pcolor   - make a pseudocolor plot
    plot     - make a line plot
    psd      - make a plot of power spectral density
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
    """
    pass




def get_current_fig_manager():
    figManager = _matlab_helpers.Gcf.get_active()
    if figManager is None:
        gcf()  # creates an active figure as a side effect
        figManager = _matlab_helpers.Gcf.get_active()
    return figManager

def mpl_connect(s, func):
    return get_current_fig_manager().canvas.mpl_connect(s, func)
mpl_connect.__doc__ = FigureCanvasBase.mpl_connect.__doc__

def mpl_disconnect(cid):
    return get_current_fig_manager().canvas.mpl_disconnect(cid)
mpl_disconnect.__doc__ = FigureCanvasBase.mpl_disconnect.__doc__

def get_plot_commands(): return ( 'axes', 'axis', 'bar', 'cla', 'clf',
    'close', 'colorbar', 'cohere', 'csd', 'draw', 'errorbar',
    'figlegend', 'figtext', 'figimage', 'figure', 'fill', 'gca',
    'gcf', 'gci', 'get', 'gray', 'barh', 'jet', 'hist', 'hold', 'imread',
    'imshow', 'legend', 'loglog', 'rc', 'pcolor', 'plot', 'psd',
    'savefig', 'scatter', 'set', 'semilogx', 'semilogy', 'show',
    'specgram', 'stem', 'subplot', 'table', 'text', 'title', 'xlabel',
    'ylabel')

def raise_msg_to_str(msg):
    """msg is a return arg from a raise.  Join with new lines"""
    if not is_string_like(msg):
        msg = '\n'.join(map(str, msg))
    return msg

#----- Now we get started with the matlab commands  ----#


# wrap the plot commands defined in axes
__fmt = """\
def %s(*args, **kwargs):
    try:
        ret =  gca().%s(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
%s.__doc__ = Axes.%s.__doc__
"""

# these methods are all simple wrappers of Axes methods by the same
# name.  We'll autogenerate these to avoid some of the boilerplate
# using the fmt string above.
_methods = (
    'axhline',
    'axhspan',
    'axvline',
    'axvspan',
    'bar',
    'barh',
    'cla',
    'cohere',
    'csd',
    'errorbar',
    'fill',
    'grid',
    'hist',
    'hlines',
    'legend',
    'loglog',
    'pcolor',
    'pcolor_classic',
    'plot',
    'plot_date',
    'psd',
    'scatter',
    'scatter_classic',
    'semilogx',
    'semilogy',
    'stem',
    'table',
    'text',
    'vlines',
    )

for name in _methods:
    exec(__fmt%(name, name, name, name))


def axis(*v):
    """\
Set/Get the axis properties::

    axis()  returns the current axis as a length a length 4 vector

    axis(v) where v = [xmin, xmax, ymin, ymax] sets the min and max of the x
        and y axis limits

    axis('off') turns off the axis lines and labels

    axis('equal') sets the xlim width and ylim height to be to be
        identical.  The longer of the two intervals is chosen
 
"""
    
    if len(v)==1 and is_string_like(v[0]):
        s = v[0]
        if s.lower()=='on': gca().set_axis_on()
        elif s.lower()=='off': gca().set_axis_off()
        elif s.lower()=='equal':
            ax = gca()
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            width = xmax-xmin
            height = ymax-ymin
            # TODO: handle decreasing lim
            
            interval = max([width, height])
            ax.set_xlim((xmin, xmin+interval))
            ax.set_ylim((ymin, ymin+interval))            
            draw_if_interactive()
            
        else:
            error_msg('Unrecognized string %s to axis; try on or off' % s)
        return
    
    try: v[0]
    except IndexError:
        xlim = gca().get_xlim()
        ylim = gca().get_ylim()
        return [xlim[0], xlim[1], ylim[0], ylim[1]]
    
    v = v[0]
    if len(v) != 4:
        error_msg('v must contain [xmin xmax ymin ymax]')
        return 
    gca().set_xlim([v[0], v[1]])
    gca().set_ylim([v[2], v[3]])
    draw_if_interactive()
    
def axes(*args, **kwargs):
    """
    Add an axes at positon rect specified by::

    axes() by itself creates a default full subplot(111) window axis

    axes(rect, axisbg='w') where rect=[left, bottom, width, height] in
    normalized (0,1) units.  axisbg is the background color for the
    axis, default white

    axes(h) where h is an axes instance makes h the
    current axis An Axes instance is returned
    """

    nargs = len(args)
    if len(args)==0: return subplot(111, **kwargs)
    if nargs>1:
        error_msg('Only one non keyword arg to axes allowed')
        return
    
    arg = args[0]

    if isinstance(arg, Axes):
        get_current_fig_manager().set_current_axes(arg)
        ret = arg
    else:
        rect = arg
        ret = get_current_fig_manager().add_axes(rect, **kwargs)
    draw_if_interactive()
    return ret


def _get_target_images(target=None):
    if target is None:
        ax = gca()
        all = ax.get_images()

        if not len(all):
            error_msg('You must first define an image, eg with imshow')
            return

        images = [all[-1]]
    else:
        if iterable(target): images = target
        else: images = [target]

    return images

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
    im = gci._current
    if im is None:
        error_msg('You must first define an image, eg with imshow')
        return
    im.set_clim(vmin, vmax)
    draw_if_interactive()
    
def close(*args):
    """
    Close a figure window

    close() by itself closes the current figure

    close(num) closes figure number num

    close(h) where h is a figure handle(instance) closes that figure

    close('all') closes all the figure windows
    """

    if len(args)==0:
        figManager = _matlab_helpers.Gcf.get_active()
        if figManager is None: return
        else: _matlab_helpers.Gcf.destroy(figManager.num)
    elif len(args)==1:
        arg = args[0]
        if arg=='all':
            for manager in _matlab_helpers.Gcf.get_all_fig_managers():
                _matlab_helpers.Gcf.destroy(manager.num)
        elif isinstance(arg, int):
            _matlab_helpers.Gcf.destroy(arg)
        elif isinstance(arg, Figure):
            for manager in _matlab_helpers.Gcf.get_all_fig_managers():
                if manager.figure==arg:
                    _matlab_helpers.Gcf.destroy(manager.num)
        else:
            msg = 'Unrecognized argument type to close'
            popup_err(msg)
    else:
        msg = 'close takes 0 or 1 arguments'
        popup_err(msg)




def clf():
    """
    Clear the current figure
    """
    manager = get_current_fig_manager()
    manager.clf()
    draw_if_interactive()

def colorbar(tickfmt='%1.1f'):
    """
    Create a colorbar for current mappable image (see gci)

    tickfmt is a format string to format the colorbar ticks

    return value is the colorbar axes instance
    """

    mappable = gci()
    if mappable is None:
        error_msg('First define a mappable image (eg imshow, figimage, pcolor, scatter')
        return

    if isinstance(mappable, image.FigureImage):
        error_msg('Colorbars for figure images currently not supported')
        return
        
    ax = gca()

    cmap = mappable.cmap
    norm = mappable.norm
    
    if norm.vmin is None or norm.vmax is None:
        mappable.autoscale()
    cmin = norm.vmin
    cmax = norm.vmax
    l,b,w,h = ax.get_position()

    neww = 0.8*w
    ax.set_position((l,b,neww,h))
    cax = axes([l + 0.9*w, b, 0.1*w, h])
    N = 200
    c = linspace(cmin, cmax, N)
    C = array([c,c])

    coll = cax.imshow(transpose(C), interpolation='nearest',
                      origin='lower',
                      cmap=cmap, norm=norm,
                      extent=(0, 1, cmin, cmax))
    mappable.add_observer(coll)
    
    cax.set_xticks([])
    cax.yaxis.tick_right()

    # restore the current axes
    axes(ax)
    return cax



def draw():
    'redraw the current figure'
    get_current_fig_manager().canvas.draw()
    
def figtext(*args, **kwargs):    
    try: ret =  gcf().text(*args, **kwargs)
    except RuntimeError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    else:
        draw_if_interactive()
        return ret
figtext.__doc__ = Figure.text.__doc__

def figimage(*args, **kwargs):    
    try: ret =  gcf().figimage(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    except RuntimeError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    draw_if_interactive()
    gci._current = ret
    return ret
figimage.__doc__ = Figure.figimage.__doc__
    
def figlegend(handles, labels, loc):
    """
    Place a legend in the figure.  Labels are a sequence of
    strings, handles is a sequence of line or patch instances, and
    loc can be a string or an integer specifying the legend
    location

    USAGE: 
      legend( (line1, line2, line3),
              ('label1', 'label2', 'label3'),
              'upper right')

    See help(legend) for information about the location codes

    A matplotlib.legend.Legend instance is returned
    """
    l=  gcf().legend(handles, labels, loc)
    draw_if_interactive()
    return l
    

def figure(num=1,
           figsize   = None, # defaults to rc figure.figsize
           dpi       = None, # defaults to rc figure.dpi
           facecolor = None, # defaults to rc figure.facecolor
           edgecolor = None, # defaults to rc figure.edgecolor
           frameon = True,
           ):
    """
    figure(num = 1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


    Create a new figure and return a handle to it

    If figure(num) already exists, make it active and return the
    handle to it.

      figure(1)

    figsize - width in height x inches; defaults to rc figure.figsize
    dpi     - resolution; defaults to rc figure.dpi
    facecolor - the background color; defaults to rc figure.facecolor
    edgecolor - the border color; defaults to rc figure.edgecolor

    rcParams gives the default values from the .matplotlibrc file

    """

    if figsize is None   : figsize   = rcParams['figure.figsize']
    if dpi is None       : dpi       = rcParams['figure.dpi']
    if facecolor is None : facecolor = rcParams['figure.facecolor']
    if edgecolor is None : edgecolor = rcParams['figure.edgecolor']


    if num==0:
        error_msg('Figure number can not be 0.\n' + \
                  'Hey, give me a break, this is matlab compatability')

    figManager = _matlab_helpers.Gcf.get_fig_manager(num)
    if figManager is None:
        if get_backend()=='PS':  dpi = 72
        figManager = new_figure_manager(num, figsize, dpi, facecolor, edgecolor, frameon)
        _matlab_helpers.Gcf.set_active(figManager)
    
    return figManager.canvas.figure

def gca():
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

    return get_current_fig_manager().get_current_axis()
        
def gcf():
    "Return a handle to the current figure"
    
    figManager = _matlab_helpers.Gcf.get_active()
    if figManager is not None:
        return figManager.canvas.figure
    else:
        return figure()

def gci():
    """
    get the current ScalarMappable instance (image or patch
    collection), or None if no images or patch collecitons have been
    defined.  The commands imshow and figimage create images
    instances, and the commands pcolor and scatter create patch
    collection instances
    """
    return gci._current
gci._current = None

def get(o, s):
    """
    Return the value of handle property s

    h is an instance of a class, eg a Line2D or an Axes or Text.
    if s is 'somename', this function returns

      o.get_somename()
    
    """
    func = 'o.get_%s()' % s
    return eval(func, {}, {'o': o})

def gray():
    'set the default colormap to gray and apply to current image if any'
    rc('image', cmap='gray')
    im = gci()
    if im is not None:
        im.set_cmap(cm.gray)
        draw_if_interactive()
        
def jet():
    'set the default colormap to jet and apply to current image if any'
    rc('image', cmap='jet')
    im = gci()
    if im is not None:
        im.set_cmap(cm.jet)
    draw_if_interactive()
    
    
def hold(b=None):
    """
    Set the hold state.  If hold is None (default), toggle the
    hold state.  Else set the hold state to boolean value b.

    Eg
    hold()      # toggle hold
    hold(True)  # hold is on
    hold(False) # hold is off
    """
    gcf().hold(b)    
    gca().hold(b)    
    rc('axes', hold=b)

    

def imread(*args, **kwargs):
    return _imread(*args, **kwargs)
imread.__doc__ = _imread.__doc__

def imshow(*args, **kwargs):
    try: im =  gca().imshow(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    except RuntimeError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    else:
        draw_if_interactive()
    gci._current = im

    return im
imshow.__doc__ = Axes.imshow.__doc__

def load(fname):
    """
    Load ASCII data from fname into an array and return the array.

    The data must be regular, same number of values in every row

    fname can be a filename or a file handle

    matfile data is not currently supported, but see
    Nigel Wade's matfile ftp://ion.le.ac.uk/matfile/matfile.tar.gz

    Example usage:

    x,y = load('test.dat')  # data in two columns

    X = load('test.dat')    # a matrix of data

    x = load('test.dat')    # a single column of data

    """

    if is_string_like(fname):
        fh = file(fname)
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    
    X = []
    numCols = None
    for line in fh:
        line = line.strip()
        if not len(line): continue
        row = [float(val) for val in line.split()]
        thisLen = len(row)
        if numCols is not None and thisLen != numCols:
            raise ValueError('All rows must have the same number of columns')
        X.append(row)

    X = array(X)
    r,c = X.shape
    if r==1 or c==1:
        X.shape = max([r,c]),
    return X

def pcolor(*args, **kwargs):
    try:
        ret =  gca().pcolor(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        gci._current = ret    
        draw_if_interactive()
        return ret
pcolor.__doc__ = Axes.pcolor.__doc__


def rc(*args, **kwargs):
    matplotlib.rc(*args, **kwargs)
    draw_if_interactive()
rc.__doc__ =   matplotlib.rc.__doc__

def rcdefaults():
    matplotlib.rcdefaults()
    draw_if_interactive()
rcdefaults.__doc__ =   matplotlib.rcdefaults.__doc__

def save(fname, X, fmt='%1.4f'):
    """
    Save the data in X to file fname using fmt string to convert the
    data to strings

    fname can be a filename or a file handle    

    Example usage:

    save('test.out', X)         # X is an array
    save('test1.out', (x,y,z))  # x,y,z equal sized 1D arrays
    save('test2.out', x)        # x is 1D
    save('test3.out', x, fmt='%1.4e')  # use exponential notation

    """

    if is_string_like(fname):
        fh = file(fname, 'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    
    X = asarray(X)
    origShape = None
    if len(X.shape)==1:
        origShape = X.shape
        X.shape = len(X), 1
    for row in X:
        fh.write(' '.join([fmt%val for val in row]) + '\n')

    if origShape is not None:
        X.shape = origShape

def savefig(*args, **kwargs):
    """
    SAVEFIG(fname, dpi=150, facecolor='w', edgecolor='w',
            orientation='portrait'):

    Save the current figure to filename fname.  dpi is the resolution
    in dots per inch.

    Output file types currently supported are jpeg and png and will be
    deduced by the extension to fname

    facecolor and edgecolor are the colors os the figure rectangle

    orientation is either 'landscape' or 'portrait' - not supported on
    all backends; currently only on postscript output."""

    for key in ('dpi', 'facecolor', 'edgecolor'):
        if not kwargs.has_key(key):
            kwargs[key] = rcParams['savefig.%s'%key]

    manager = get_current_fig_manager()
    manager.canvas.print_figure(*args, **kwargs)


    
def set(h, *args, **kwargs):
    """
    Set handle h property in string s to value val

    h can be a handle or vector of handles.

    h is an instance (or vector of instances) of a class, eg a Line2D
    or an Axes or Text.

    args is a list of string, value pairs.  if the string
    is 'somename', set function calls

      o.set_somename(value)

    for every instance in h.

    
    """

    if not iterable(h): h = [h]
    else: h = flatten(h)

    if len(args)%2:
        error_msg('The set args must be string, value pairs')

    funcvals = []
    for i in range(0, len(args)-1, 2):
        funcvals.append((args[i], args[i+1]))
    funcvals.extend(kwargs.items())

    ret = []
    for o in h:
        for s, val in funcvals:
            s = s.lower()
            funcName = "set_%s"%s
            func = getattr(o,funcName)        
            try: ret.extend( [func(val)] )
            except ValueError, msg:
                msg = exception_to_str(msg)
                error_msg(msg)
                raise RuntimeError(msg)
        
    draw_if_interactive()
    return [x for x in flatten(ret)]

def specgram(*args, **kwargs):
    try: ret =  gca().specgram(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        Pxx, freqs, bins, im = ret
        gci._current = im
        draw_if_interactive()
        return ret
specgram.__doc__ = Axes.specgram.__doc__

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
    """
    try:
        get_current_fig_manager().add_subplot(*args, **kwargs)
        a =  gca()
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError, msg
    draw_if_interactive()
    return a


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
    """
    ax = gca()
    if len(args)==0: return ax.get_xlim()
    elif len(args)==1: lim = ax.set_xlim(args)
    elif len(args)==2: lim = ax.set_xlim((args[0], args[1]))
    else: raise RuntimeError('Illegal number of arguments to xlim')
    draw_if_interactive()
    return lim

def ylim(*args, **kwargs):
    """
    Set/Get the ylimits of the current axes

    ymin, ymax = ylim()   : return the current ylim
    ylim( (ymin, ymax) )  : set the ylim to ymin, ymax
    ylim( ymin, ymax )    : set the ylim to ymin, ymax    
    """
    ax = gca()
    if len(args)==0: return ax.get_ylim()
    elif len(args)==1: lim = ax.set_ylim(args)
    elif len(args)==2: lim = ax.set_ylim((args[0], args[1]))
    else: raise RuntimeError('Illegal number of arguments to ylim')
    draw_if_interactive()
    return lim

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
        labels = ax.set_xticklabels(args[1])
    else: raise RuntimeError('Illegal number of arguments to xticks')
    draw_if_interactive()
    return locs, labels

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
        labels = ax.set_yticklabels(args[1])
    else: raise RuntimeError('Illegal number of arguments to yticks')
    draw_if_interactive()
    return locs, labels


