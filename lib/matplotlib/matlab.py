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

def bar(*args, **kwargs):
    try: patches =  gca().bar(*args, **kwargs)
    except Exception, msg:
        s = exception_to_str(msg)
        error_msg(s)
        raise RuntimeError(msg)
    draw_if_interactive()
    return patches
bar.__doc__ = Axes.bar.__doc__

def barh(*args, **kwargs):
    try: patches =  gca().barh(*args, **kwargs)
    except Exception, msg:
        s = exception_to_str(msg)
        error_msg(s)
        raise RuntimeError(msg)
    draw_if_interactive()
    return patches
barh.__doc__ = Axes.barh.__doc__

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



def errorbar(*args, **kwargs):
    try: ret =  gca().errorbar(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
errorbar.__doc__ = Axes.errorbar.__doc__
    
def cla():
    """
    Clear the current axes
    """
    gca().cla()
    draw_if_interactive()
    
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

def cohere(x, y, NFFT=256, Fs=2, detrend=mlab.detrend_none,
           window=mlab.window_hanning, noverlap=0):
    """
    Compute the coherence between x and y.  Coherence is the
    normalized cross spectral density

    Cxy = |Pxy|^2/(Pxx*Pyy)

    The return value is (Cxy, f), where f are the frequencies of the
    coherence vector.  See the docs for psd and csd for information
    about the function arguments NFFT, detrend, windowm noverlap, as
    well as the methods used to compute Pxy, Pxx and Pyy.

    Returns the tuple Cxy, freqs

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    try: ret =  gca().cohere(x, y, NFFT, Fs, detrend, window, noverlap)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret


def csd(x, y, NFFT=256, Fs=2, detrend=mlab.detrend_none,
        window=mlab.window_hanning, noverlap=0):
    """
    The cross spectral density Pxy by Welches average periodogram
    method.  The vectors x and y are divided into NFFT length
    segments.  Each segment is detrended by function detrend and
    windowed by function window.  noverlap gives the length of the
    overlap between segments.  The product of the direct FFTs of x and
    y are averaged over each segment to compute Pxy, with a scaling to
    correct for power loss due to windowing.  Fs is the sampling
    frequency.

    NFFT must be a power of 2

    detrend and window are functions, unlike in matlab where they are
    vectors.  For detrending you can use detrend_none, detrend_mean,
    detrend_linear or a custom function.  For windowing, you can use
    window_none, window_hanning, or a custom function

    Returns the tuple Pxy, freqs.  Pxy is the cross spectrum (complex
    valued), and 10*log10(|Pxy|) is plotted

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """
    try: ret =  gca().csd(x, y, NFFT, Fs, detrend, window, noverlap)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret


def draw():
    'redraw the current figure'
    get_current_fig_manager().canvas.draw()
    
def figtext(*args, **kwargs):    
    try: t =  gcf().text(*args, **kwargs)
    except RuntimeError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    else:
        draw_if_interactive()
        return t
figtext.__doc__ = Figure.text.__doc__

def figimage(*args, **kwargs):    
    try: im =  gcf().figimage(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    except RuntimeError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    draw_if_interactive()
    gci._current = im
    return im
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

def fill(*args, **kwargs):
    """
    plot filled polygons.  *args is a variable length argument,
    allowing for multiple x,y pairs with an optional color format
    string.  For example, all of the following are legal, assuming a
    is the Axis instance:

      fill(x,y)            # plot polygon with vertices at x,y
      fill(x,y, 'b' )      # plot polygon with vertices at x,y in blue

    An arbitrary number of x, y, color groups can be specified, as in 
      fill(x1, y1, 'g', x2, y2, 'r')  

    Return value is a list of patches that were added

    The following color strings are supported

      b  : blue
      g  : green
      r  : red
      c  : cyan
      m  : magenta
      y  : yellow
      k  : black 
      w  : white

    The kwargs that are can be used to set line properties (any
    property that has a set_* method).  You can use this to set edge
    color, face color, etc.

    Example code:

    from matplotlib.matlab import *
    t = arange(0.0, 1.01, 0.01)
    s = sin(2*2*pi*t)

    fill(t, s, 'r')
    grid(True)
    show()

    """
    try: patch =  gca().fill(*args, **kwargs)
    except Exception, msg:
        s = exception_to_str(msg)
        error_msg(s)
        raise RuntimeError('Could not exectute fill')
    draw_if_interactive()
    return patch


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
    

def grid(b=None):
    """
    Set the figure grid to be on or off (b is a boolean)

    if b is None, toggle the grid state
    """
    
    gca().grid(b)
    draw_if_interactive()

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
    
def hist(x, bins=10, noplot=0, normed=0, bottom=0):
    """
    Compute the histogram of x.  bins is either an integer number of
    bins or a sequence giving the bins.  x are the data to be binned.

    if noplot is True, just compute the histogram and return the
    number of observations and the bins as an (n, bins) tuple.

    If noplot is False, compute the histogram and plot it, returning
    n, bins, patches

    If normed is true, the first element of the return tuple will be the
    counts normalized to form a probability distribtion, ie,
    n/(len(x)*dbin)
    

    To control the properties of the returned patches, you can can
    call any of the patch methods on those patches; see
    matplotlib.patches and matplotlib.artist (the base class for
    patches).  Eg

    n, bins, patches = hist(x, 50, normed=1)
    set(patches, 'facecolor', 'g', 'alpha', 0.75)

    """
    if noplot: return mlab.hist(x, bins, normed)
    else:
        try:
            ret = gca().hist(x, bins, normed, bottom)
        except ValueError, msg:
            msg = raise_msg_to_str(msg)
            error_msg(msg)
            raise RuntimeError, msg
    draw_if_interactive()
    return ret

    
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

    
def hlines(*args, **kwargs):    
    """
    lines = hlines(self, y, xmin, xmax, fmt='k-')

    plot horizontal lines at each y from xmin to xmax.  xmin or
    xmax can be scalars or len(x) numpy arrays.  If they are
    scalars, then the respective values are constant, else the
    widths of the lines are determined by xmin and xmax

    Returns a list of line instances that were added

    """
    try: lines =  gca().hlines(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError, msg
    draw_if_interactive()
    return lines

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

def legend(*args, **kwargs):
    """
    Place a legend on the current axes at location loc.  Labels are a
    sequence of strings and loc can be a string or an integer
    specifying the legend location

    USAGE: 

      Make a legend with existing lines
      legend( LABELS )
      >>> legend( ('label1', 'label2', 'label3') ) 

      Make a legend for Line2D instances lines1, line2, line3
      legend( LINES, LABELS )
      >>> legend( (line1, line2, line3), ('label1', 'label2', 'label3') )

      Make a legend at LOC
      legend( LABELS, LOC )  or
      legend( LINES, LABELS, LOC )
      >>> legend( ('label1', 'label2', 'label3'), loc='upper left')
      >>> legend( (line1, line2, line3),
                  ('label1', 'label2', 'label3'),
                  loc=2)

    The LOC location codes are

    The LOC location codes are

      'best' : 0,          (currently not supported, defaults to upper right)
      'upper right'  : 1,  (default)
      'upper left'   : 2,
      'lower left'   : 3,
      'lower right'  : 4,
      'right'        : 5,
      'center left'  : 6,
      'center right' : 7,
      'lower center' : 8,
      'upper center' : 9,
      'center'       : 10,

    If none of these are suitable, loc can be a 2-tuple giving x,y
    in axes coords, ie,

      loc = 0, 1 is left top
      loc = 0.5, 0.5 is center, center

      and so on


    The legend instance is returned
    """

    ret = gca().legend(*args, **kwargs)
    draw_if_interactive()
    return ret

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


    
            

def loglog(*args, **kwargs):
    try: ret =  gca().loglog(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
loglog.__doc__ = Axes.loglog.__doc__

def pcolor(*args, **kwargs):
    ret = gca().pcolor(*args, **kwargs)
    gci._current = ret    
    draw_if_interactive()
    return ret
pcolor.__doc__ = Axes.pcolor.__doc__

def pcolor_classic(*args, **kwargs):
    """
    pcolor_classic(C) - make a pseudocolor plot of matrix C

    pcolor_classic(X, Y, C) - a pseudo color plot of C on the matrices X and Y  

    Shading:

      The optional keyword arg shading ('flat' or 'faceted') will
      determine whether the black grid is drawn around each pcolor
      square.  Defaul 'faceteted'
         e.g.,   
         pcolor_classic(C, shading='flat')  
         pcolor_classic(X, Y, C, shading='faceted')

    returns a list of patch objects.


    pcolor(C, cmap=cm.jet) - make a pseudocolor plot of matrix C using
    rectangle patches using a colormap jet.  Colormaps are avalible in
    matplotlib.cm.  You must pass this as a kwarg.
        
    pcolor(C, norm=normalize()) - the normalization function used to
    scale your color data to 0-1.  must be passed as a kwarg.
    normalization functions are derived from matplotlib.colors.Norm

    pcolor(C, alpha=0.5) - set the alpha of the pseudocolor plot.
    Must be used as a kwarg

    Note, the behavior of meshgrid in matlab is a bit
    counterintuitive for x and y arrays.  For example,

      x = arange(7)
      y = arange(5)
      X, Y = meshgrid(x,y)

      Z = rand( len(x), len(y))
      pcolor(X, Y, Z)

    will fail in matlab and matplotlib.  You will probably be
    happy with

      pcolor_classic(X, Y, transpose(Z))

    Likewise, for nonsquare Z,

      pcolor_classic(transpose(Z))

    will make the x and y axes in the plot agree with the numrows
    and numcols of Z

    """
    ret = gca().pcolor_classic(*args, **kwargs)
    draw_if_interactive()
    return ret

def plot(*args, **kwargs):
    try:
        lines =  gca().plot(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return lines
plot.__doc__ = Axes.plot.__doc__


def plot_date(*args, **kwargs):
    try: lines =  gca().plot_date(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return lines
plot_date.__doc__ = Axes.plot_date.__doc__

def psd(x, NFFT=256, Fs=2, detrend=mlab.detrend_none,
        window=mlab.window_hanning, noverlap=0):
    """
    The power spectral density by Welches average periodogram method.
    The vector x is divided into NFFT length segments.  Each segment
    is detrended by function detrend and windowed by function window.
    noperlap gives the length of the overlap between segments.  The
    absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
    with a scaling to correct for power loss due to windowing.  Fs is
    the sampling frequency.

    -- NFFT must be a power of 2

    -- detrend and window are functions, unlike in matlab where they
       are vectors.  For detrending you can use detrend_none,
       detrend_mean, detrend_linear or a custom function.  For
       windowing, you can use window_none, window_hanning, or a custom
       function

    -- if length x < NFFT, it will be zero padded to NFFT
    

    Returns the tuple Pxx, freqs

    For plotting, the power is plotted as 10*log10(pxx)) for decibels,
    though pxx itself is returned
    
    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """
    try: ret =  gca().psd(x, NFFT, Fs, detrend, window, noverlap)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
    

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
    def savefig(fname, dpi=150, facecolor='w', edgecolor='w',
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

def scatter(*args, **kwargs):
    try: coll =  gca().scatter(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError, msg
    
    draw_if_interactive()
    gci._current = coll
    return coll
scatter.__doc__ = Axes.scatter.__doc__

def scatter_classic(*args, **kwargs):
    """

    scatter_classic(self, x, y, s=None, c='b'):

    Make a scatter plot of x versus y.  s is a size (in data
    coords) and can be either a scalar or an array of the same
    length as x or y.  c is a color and can be a single color
    format string or an length(x) array of intensities which will
    be mapped by the colormap jet.        

    If size is None a default size will be used
    """

    try: patches =  gca().scatter_classic(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError, msg
    draw_if_interactive()
    return patches



def semilogx(*args, **kwargs):

    try: ret =  gca().semilogx(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
semilogx.__doc__ = Axes.semilogx.__doc__

def semilogy(*args, **kwargs):
    try: ret =  gca().semilogy(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
semilogy.__doc__ = Axes.semilogy.__doc__        
    
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

def stem(*args, **kwargs):
    """
    stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')

    A stem plot plots vertical lines (using linefmt) at each x
    location from the baseline to y, and places a marker there using
    markerfmt.  A horizontal line at 0 is is plotted using basefmt
    
    return value is markerline, stemlines, baseline

    See
    http://www.mathworks.com/access/helpdesk/help/techdoc/ref/stem.html
    for details and examples/stem_plot.py for a demo.
    """
    try: ret =  gca().stem(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
    

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

def text(x, y, label, fontdict=None, **kwargs):
    """
    Add text to axis at location x,y

    fontdict is a dictionary to override the default text properties.
    If fontdict is None, the default is

      'fontsize'            : 'x-small',
      'verticalalignment'   : 'bottom',
      'horizontalalignment' : 'left'

    **kwargs can in turn be used to override the fontdict, as in

      a.text(x,y,label, fontsize='medium')

    This command supplies no override dict, and so will have
    'verticalalignment'='bottom' and 'horizontalalignment'='left' but
    the keyword arg 'fontsize' will create a fontsize of medium or 12

    The purpose these options is to make it easy for you to create a
    default font theme for your plots by creating a single dictionary,
    and then being able to selective change individual attributes for
    the varous text creation commands, as in

        fonts = {
          'color'               : 'k',
          'fontname'            : 'Courier',
          'fontweight'          : 'bold'
          }

        title('My title', fonts, fontsize='medium')
        xlabel('My xlabel', fonts, fontsize='small')
        ylabel('My ylabel', fonts, fontsize='small')
        text(12, 20, 'some text', fonts, fontsize='x-small')

    The Text defaults are

        'color'               : 'k',
        'fontname'            : 'Sans',
        'fontsize'            : 'small',
        'fontweight'          : 'bold',
        'fontangle'           : 'normal',
        'horizontalalignment' : 'left'
        'rotation'            : 'horizontal',
        'verticalalignment'   : 'bottom',
        'transx'              : gca().xaxis.transData,
        'transy'              : gca().yaxis.transData,            

        transx and transy specify that text is in data coords,
        alternatively, you can specify text in axis coords (0,0 lower
        left and 1,1 upper right).  The example below places text in
        the center of the axes

        ax = subplot(111)
        text(0.5, 0.5,'matplotlib', 
             horizontalalignment='center',
             verticalalignment='center',
             transx = ax.xaxis.transAxis,
             transy = ax.yaxis.transAxis,
        )

    """
    t =  gca().text(x, y, label, fontdict, **kwargs)
    draw_if_interactive()
    return t

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



def vlines(*args, **kwargs):    
    """
    lines =  vlines(x, ymin, ymax, color='k'):

    Plot vertical lines at each x from ymin to ymax.  ymin or ymax
    can be scalars or len(x) numpy arrays.  If they are scalars,
    then the respective values are constant, else the heights of
    the lines are determined by ymin and ymax

    Returns a list of lines that were added
    """
    try: lines =  gca().vlines(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError, msg
    else:  draw_if_interactive()
    return lines

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


def table(*args, **kwargs):
    """
    table(cellText=None, cellColours=None,
          cellLoc='right', colWidths=None,
          rowLabels=None, rowColours=None, rowLoc='left',
          colLabels=None, colColours=None, colLoc='center',
          loc='bottom', bbox=None):

    Add a table to the current axes.  Returns a table instance.  For
    finer grained control over tables, use the Table class and add it
    to the axes with add_table.

    Thanks to John Gill for providing the class and table.

    """
    t = gca().table(*args, **kwargs)
    draw_if_interactive()
    return t
    


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

def axhline(y=0, xmin=0, xmax=1, **kwargs):
    """\
axhline : Axis Horizontal Line

Draw a horizontal line at y from xmin to xmax.  With the default
values of xmin=0 and xmax=1, this line will always span the horizontal
extent of the axes, regardless of the xlim settings, even if you
change them, eg with the xlim command.  That is, the horizontal extent
is in axes coords: 0=left, 0.5=middle, 1.0=right but the y location is
in data coordinates.

return value is the Line2D instance.  kwargs are the same as kwargs to
plot, and can be used to control the line properties.  Eg

  # draw a thick red hline at y=0 that spans the xrange
  l = axhline(linewidth=4, color='r')

  # draw a default hline at y=1 that spans the xrange
  l = axhline(y=1)

  # draw a default hline at y=.5 that spans the the middle half of
  # the xrange
  l = axhline(y=.5, xmin=0.25, xmax=0.75)
  ylim(-1,2)
    """
    ax = gca()
    trans = blend_xy_sep_transform( ax.transAxes, ax.transData)
    
    l, = ax.plot([xmin,xmax], [y,y], transform=trans, **kwargs)
    draw_if_interactive()
    return l


def axvline(x=0, ymin=0, ymax=1, **kwargs):
    """\
axvline : Axis Vertical Line

Draw a vertical line at x from ymin to ymax.  With the default values
of ymin=0 and ymax=1, this line will always span the vertical extent
of the axes, regardless of the xlim settings, even if you change them,
eg with the xlim command.  That is, the vertical extent is in axes
coords: 0=bottom, 0.5=middle, 1.0=top but the x location is in data
coordinates.

return value is the Line2D instance.  kwargs are the same as
kwargs to plot, and can be used to control the line properties.  Eg

# draw a thick red vline at x=0 that spans the yrange
l = axvline(linewidth=4, color='r')

# draw a default vline at x=1 that spans the yrange
l = axvline(x=1)

# draw a default vline at x=.5 that spans the the middle half of
# the yrange
l = axvline(x=.5, ymin=0.25, ymax=0.75)
xlim(-1,2)
    """
    ax = gca()
    trans = blend_xy_sep_transform( ax.transData, ax.transAxes )
    
    l, = ax.plot([x,x], [ymin,ymax] , transform=trans, **kwargs)
    draw_if_interactive()
    return l


def axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs):
    """\
axhspan : Axis Horizontal Span.  ycoords are in data units and x
coords are in axes (relative 0-1) units

Draw a horizontal span (regtangle) from ymin to ymax.  With the
default values of xmin=0 and xmax=1, this always span the xrange,
regardless of the xlim settings, even if you change them, eg with the
xlim command.  That is, the horizontal extent is in axes coords:
0=left, 0.5=middle, 1.0=right but the y location is in data
coordinates.

kwargs are the kwargs to Patch, eg

  antialiased, aa
  linewidth,   lw
  edgecolor,   ec
  facecolor,   fc

the terms on the right are aliases

return value is the patches.Polygon instance.

#draws a gray rectangle from y=0.25-0.75 that spans the horizontal
#extent of the axes
p = axhspan(0.25, 0.75, facecolor=0.5, alpha=0.5)


"""
    ax = gca()
    trans = blend_xy_sep_transform( ax.transAxes, ax.transData  )
    verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)

    p = Polygon(verts, **kwargs)
    p.set_transform(trans)
    ax.add_patch(p)
    draw_if_interactive()
    return p


def axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs):
    """\
axvspan : Axis Vertical Span.  xcoords are in data units and y coords
are in axes (relative 0-1) units

Draw a vertical span (regtangle) from xmin to xmax.  With the default
values of ymin=0 and ymax=1, this always span the yrange, regardless
of the ylim settings, even if you change them, eg with the ylim
command.  That is, the vertical extent is in axes coords: 0=bottom,
0.5=middle, 1.0=top but the y location is in data coordinates.

kwargs are the kwargs to Patch, eg

  antialiased, aa
  linewidth,   lw
  edgecolor,   ec
  facecolor,   fc

the terms on the right are aliases

return value is the patches.Polygon instance.

# draw a vertical green translucent rectangle from x=1.25 to 1.55 that
# spans the yrange of the axes
p = axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

"""
    ax = gca()
    trans = blend_xy_sep_transform( ax.transData, ax.transAxes   )
    verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)

    p = Polygon(verts, **kwargs)
    p.set_transform(trans)
    ax.add_patch(p)
    draw_if_interactive()
    return p
