"""
This is a matlab(TM) style interface to matplotlib.

The following plotting commands are provided; some of these do not
exist in matlab(TM) but have proven themselves to be useful nonetheless.
The majority of them, however, have matlab analogs

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
  clim     - adjust the color limits of the current image
  close    - close a figure window
  colorbar - add a colorbar to the current figure
  cohere   - make a plot of coherence
  contour  - make a contour plot
  csd      - make a plot of cross spectral density  
  delaxes  - delete an axes from the current figure
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
  grid     - set whether gridding is on
  hist     - make a histogram
  hold     - set the axes hold state
  ioff     - turn interaction mode off
  ion      - turn interaction mode on
  isinteractive - return True if interaction mode is on
  imread   - load image file into array
  imshow   - plot image data
  ishold   - return the hold state of the current axes
  legend   - make an axes legend
  loglog   - a log log plot
  pcolor   - make a pseudocolor plot
  pie      - make a pie chart
  plot     - make a line plot
  pie      - pie charts
  polar    - make a polar plot on a PolarAxes
  psd      - make a plot of power spectral density
  rc       - control the default params
  rgrids   - customize the radial grids and labels for polar 
  savefig  - save the current figure
  scatter  - make a scatter plot
  set      - set a handle graphics property
  semilogx - log x axis
  semilogy - log y axis
  show     - show the figures
  specgram - a spectrogram plot
  spy      - plot sparsity pattern using markers
  spy2     - plot sparsity pattern using image
  stem     - make a stem plot
  subplot  - make a subplot (numrows, numcols, axesnum)
  table    - add a table to the plot
  text     - add some text at location x,y to the current axes
  thetagrids - customize the radial theta grids and labels for polar 
  title    - add a title to the current axes
  xlim     - set/get the xlimits
  ylim     - set/get the ylimits
  xticks   - set/get the xticks
  yticks   - set/get the yticks
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

_Event handling

  connect - register an event handler
  disconnect - remove a connected event handler
  
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
  amax       - the maximum along dimension m
  mean      - the mean along dimension m
  median    - the median along dimension m
  amin       - the minimum along dimension m
  norm      - the norm of vector x
  prod      - the product along dimension m
  ptp       - the max-min along dimension m
  std       - the standard deviation along dimension m
  asum       - the sum along dimension m

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

Most of the other commands are from Numeric, MLab and FFT, with the
exception of those in mlab.py provided by matplotlib.
"""


import cm
import _pylab_helpers
import mlab  #so I can override hist, psd, etc...

from axes import Axes, PolarAxes
import backends
from backends import new_figure_manager, error_msg, \
     draw_if_interactive, show

from cbook import flatten, is_string_like, exception_to_str, popd, silent_list, iterable
from colors import normalize
from cm import ColormapJet, Grayscale, get_cmap
from figure import Figure
import image
from matplotlib import rcParams, rcParamsDefault, get_backend
from backend_bases import FigureCanvasBase
from image import imread as _imread
from lines import Line2D
from text import Text
from patches import Polygon, Rectangle, Circle
from transforms import blend_xy_sep_transform

# catch more than an import error here, since the src could fail too,
# eg a bad pytz install.  I don't want to break all of matplotlib for
# date support
try:
    from dates import date2num, num2date, drange, epoch2num, num2epoch, mx2num,\
            DateFormatter, IndexDateFormatter, DateLocator,\
            RRuleLocator, YearLocator, MonthLocator, WeekdayLocator,\
            DayLocator, HourLocator, MinuteLocator, SecondLocator,\
            rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY, MONTHLY,\
            WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY, relativedelta
except:
    __dates_all__ = []
    pass
else:
    import dates
    __dates_all__ = dates.__all__

from ticker import TickHelper, Formatter, FixedFormatter, NullFormatter,\
           FuncFormatter, FormatStrFormatter, ScalarFormatter,\
           LogFormatter, LogFormatterExponent, LogFormatterMathtext,\
           Locator, IndexLocator, FixedLocator, NullLocator,\
           LinearLocator, LogLocator, AutoLocator, MultipleLocator
import ticker
import matplotlib

# bring all the  symbols in so folks can import them from
# pylab in one fell swoop

from numerix import array, zeros, shape, rank, size, fromstring, take, put, putmask, reshape, repeat, choose, searchsorted, asum, cumsum, product, cumproduct, alltrue, sometrue, allclose, arrayrange, arange, asarray, convolve, swapaxes, concatenate, transpose, sort, argsort, argmax, argmin, innerproduct, dot, outerproduct, resize, indices, fromfunction, diagonal, trace, ravel, nonzero, shape, where, compress, clip, zeros, ones, identity, add, logical_or, exp, subtract, logical_xor, log, multiply, logical_not, log10, divide, maximum, sin, minimum, sinh, conjugate, bitwise_and, sqrt, power, bitwise_or, tan, absolute, bitwise_xor, tanh, negative, ceil, greater, fabs, greater_equal, floor, less, arccos, arctan2, less_equal, arcsin, fmod, equal, arctan, hypot, not_equal, cos, around, logical_and, cosh, arccosh, arcsinh, arctanh, cross_correlate, \
     pi, ArrayType, matrixmultiply

from numerix import Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64, Complex32, Complex64, Float, Int, Complex

from matplotlib.numerix.fft import fft
from matplotlib.numerix.linear_algebra import inverse, eigenvectors

from matplotlib.numerix.mlab import rand,randn,eye,tri,diag,fliplr,flipud,rot90,tril,triu,ptp,mean,msort,median,std,cumsum,prod,cumprod,trapz,diff,cov,corrcoef,squeeze,kaiser,blackman,bartlett,hanning,hamming,sinc,eig,svd,angle,roots,amin,  amax

from matplotlib.mlab import linspace, window_hanning, window_none, conv, detrend, detrend_mean, detrend_none, detrend_linear, corrcoef, polyfit, polyval, vander, entropy, normpdf, levypdf, find, trapz, prepca, fix, rem, norm, orth, rank, sqrtm, prctile, center_matrix, meshgrid, rk4, exp_safe, amap, sum_flat, mean_flat, rms_flat, l1norm, l2norm, norm, frange, diagonal_matrix, base_repr, binary_repr, log2, ispower2, bivariate_normal

    
    

"""
problem syms
 - cross_correlate  - getting from convolve
average
sarray
dump
dumps
load
loads
divide_safe
invert
left_shift
right_shift
sign
"""

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
    contour  - make a contour plot
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

    """
    pass

def colormaps():
    """
    matplotlib provides the following colormaps.  

      autumn bone cool copper flag gray hot hsv jet pink prism
      spring summer winter

    You can set the colormap for an image, pcolor, scatter, etc,
    either as a keyword argument

    >>> imshow(X, cmap=cm.hot)

    or post-hoc using the corresponding pylab interface function

    >>> imshow(X)
    >>> hot()
    >>> jet()

    In interactive mode, this will update the colormap allowing you to
    see which one works best for your data.
    """
    pass


def get_current_fig_manager():
    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is None:
        gcf()  # creates an active figure as a side effect
        figManager = _pylab_helpers.Gcf.get_active()
    return figManager

def connect(s, func):
    return get_current_fig_manager().canvas.mpl_connect(s, func)
connect.__doc__ = FigureCanvasBase.mpl_connect.__doc__

def disconnect(cid):
    return get_current_fig_manager().canvas.mpl_disconnect(cid)
disconnect.__doc__ = FigureCanvasBase.mpl_disconnect.__doc__

def get_plot_commands(): return ( 'axes', 'axis', 'bar', 'cla', 'clf',
    'close', 'colorbar', 'cohere', 'csd', 'draw', 'errorbar',
    'figlegend', 'figtext', 'figimage', 'figure', 'fill', 'gca',
    'gcf', 'gci', 'get', 'gray', 'barh', 'jet', 'hist', 'hold', 'imread',
    'imshow', 'legend', 'loglog', 'rc', 'pcolor', 'plot', 'psd',
    'savefig', 'scatter', 'set', 'semilogx', 'semilogy', 'show',
    'specgram', 'stem', 'subplot', 'table', 'text', 'title', 'xlabel',
    'ylabel', 'pie', 'polar')

def raise_msg_to_str(msg):
    """msg is a return arg from a raise.  Join with new lines"""
    if not is_string_like(msg):
        msg = '\n'.join(map(str, msg))
    return msg

#----- Now we get started with the matlab(TM) commands  ----#




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

kwargs:

  axisbg=color   : the axes background color
  frameon=False  : don't display the frame
  sharex=otherax : the current axes shares xaxis attribute with otherax
  sharey=otherax : the current axes shares yaxis attribute with otherax

Examples

  examples/axes_demo.py places custom axes.
  examples/shared_axis_demo.py uses sharex and sharey


    
    """

    nargs = len(args)
    if len(args)==0: return subplot(111, **kwargs)
    if nargs>1:
        error_msg('Only one non keyword arg to axes allowed')
        return
    
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
    im = gci()
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
            error_msg('Unrecognized argument type %s to close'%type(arg))
    else:
        error_msg('close takes 0 or 1 arguments')


def clf():
    """
    Clear the current figure
    """
    gcf().clf()
    draw_if_interactive()

def colorbar(tickfmt='%1.1f', cax=None, orientation='vertical'):
    """
    Create a colorbar for current mappable image (see gci)

    tickfmt is a format string to format the colorbar ticks

    cax is a colorbar axes instance in which the colorbar will be
    placed.  If None, as default axesd will be created resizing the
    current aqxes to make room for it.  If not None, the supplied axes
    will be used and the other axes positions will be unchanged.

    orientation is the colorbar orientation: one of 'vertical' | 'horizontal'
    return value is the colorbar axes instance
    """

    if orientation not in ('horizontal', 'vertical'):
        raise ValueError('Orientation must be horizontal or vertical')
    
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

    if cax is None:
        l,b,w,h = ax.get_position()
        if orientation=='vertical':
            neww = 0.8*w
            ax.set_position((l,b,neww,h))
            cax = axes([l + 0.9*w, b, 0.1*w, h])
        else:
            newh = 0.8*h
            ax.set_position((l,b+0.2*h,w,newh))
            cax = axes([l, b, w, 0.1*h])

    else:
        if not isinstance(cax, Axes):
            raise TypeError('Expected an Axes instance for cax')
        
    N = cmap.N

    c = linspace(cmin, cmax, N)
    C = array([c,c])

    if orientation=='vertical':
        C = transpose(C)

    coll = cax.imshow(C,
                      interpolation='nearest', 
                      origin='lower',
                      cmap=cmap, norm=norm,
                      extent=(0, 1, cmin, cmax))
    mappable.add_observer(coll)
    mappable.set_colorbar(coll, cax)
    if orientation=='vertical':
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.yaxis.set_major_formatter(FormatStrFormatter(tickfmt))
    else:
        cax.set_yticks([])
        cax.xaxis.set_major_formatter(FormatStrFormatter(tickfmt))
    
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
    # allow callers to override the hold state by passing hold=True|False
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
figimage.__doc__ = Figure.figimage.__doc__ + """
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

def savefig(*args, **kwargs):    
    try: ret =  gcf().savefig(*args, **kwargs)
    except RuntimeError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError(msg)
    else: return ret
savefig.__doc__ = Figure.savefig.__doc__
    

def figure(num=None, # autoincrement if None, else integer from 1-N
           figsize   = None, # defaults to rc figure.figsize
           dpi       = None, # defaults to rc figure.dpi
           facecolor = None, # defaults to rc figure.facecolor
           edgecolor = None, # defaults to rc figure.edgecolor
           frameon = True,
           ):
    """
    figure(num = None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


    Create a new figure and return a handle to it.  If num=None, the
    figure number will be incremented and a new figure will be
    created.


    If num is an integer, and figure(num) already exists, make it
    active and return the handle to it.  If figure(num) does not exist
    it will be created.  Numbering starts at 1, matlab style

      figure(1)


    kwargs:

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
                  'Hey, give me a break, this is matlab(TM) compatability')
        return 


    if num is None:
        allnums = [f.num for f in _pylab_helpers.Gcf.get_all_fig_managers()]
        if allnums:
            num = max(allnums) + 1
        else:
            num = 1

    figManager = _pylab_helpers.Gcf.get_fig_manager(num)
    if figManager is None:
        if get_backend()=='PS':  dpi = 72
        figManager = new_figure_manager(num, figsize, dpi, facecolor, edgecolor, frameon)
        _pylab_helpers.Gcf.set_active(figManager)
    
    return figManager.canvas.figure

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

    return gcf().gca(**kwargs)
        
def gcf():
    "Return a handle to the current figure"
    
    figManager = _pylab_helpers.Gcf.get_active()
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

def get(o, *args):
    """
    Return the value of handle property s

    h is an instance of a class, eg a Line2D or an Axes or Text.
    if s is 'somename', this function returns

      o.get_somename()

    get can be used to query all the gettable properties with get(o)
    Many properties have aliases for shorter typing, eg 'lw' is an
    alias for 'linewidth'.  In the output, aliases and full property
    names will be listed as

      property or  alias = value

    eg

      linewidth or lw = 2
    """

    insp = _ObjectInspector(o)

    if len(args)==0:
        print '\n'.join(insp.pprint_getters())
        return
    
    name = args[0]
    func = getattr(o, 'get_' + name)
    return func()

    
    
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

def ishold():
    """
    Return the hold status of the current axes
    """
    return gca().ishold()    

def isinteractive():
    """
    Return the interactive status
    """
    return matplotlib.is_interactive()

def imread(*args, **kwargs):
    return _imread(*args, **kwargs)
imread.__doc__ = _imread.__doc__


def load(fname,comments='%'):
    """
    Load ASCII data from fname into an array and return the array.

    The data must be regular, same number of values in every row

    fname can be a filename or a file handle

    matfile data is not currently supported, but see
    Nigel Wade's matfile ftp://ion.le.ac.uk/matfile/matfile.tar.gz

    Example usage:

    X = load('test.dat')  # data in two columns
    t = X[:,0]
    y = X[:,1]

    Alternatively, you can do

    t,y = transpose(load('test.dat')) # for  two column data
    

    X = load('test.dat')    # a matrix of data

    x = load('test.dat')    # a single column of data

    comments is the character used to indicate the start of a comment
    in the file

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
        line = line[:line.find(comments)].strip()
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


def rc(*args, **kwargs):
    matplotlib.rc(*args, **kwargs)
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


class _ObjectInspector:

    def __init__(self, o):
        if iterable(o): o = o[0]
        self.o = o
        self.aliasd = self.get_aliases()

    def get_aliases(self):
        'get a dict mapping fullname -> alias for each alias in o'
        names = [name for name in dir(self.o) if
                 (name.startswith('set_') or name.startswith('get_'))
                 and callable(getattr(self.o,name))]
        aliases = {}
        for name in names:
            func = getattr(self.o, name)
            if not self.is_alias(func): continue
            docstring = func.__doc__
            fullname = docstring[10:]
            aliases[fullname[4:]] = name[4:]
        return aliases

    def get_valid_values(self, attr):
        """
        get the legal arguments for the setter associated with attr

        This is done by querying the doc string of the function set_attr
        for a line that begins with ACCEPTS:
        """

        name = 'set_%s'%attr
        if not hasattr(self.o, name):
            raise AttributeError('%s has no function %s'%(self.o,name))
        func = getattr(self.o, name)

        docstring = func.__doc__
        if docstring is None: return 'unknown'

        if docstring.startswith('alias for '):
            return None
        for line in docstring.split('\n'):
            line = line.lstrip()
            if not line.startswith('ACCEPTS:'): continue
            return line[8:].strip()
        return 'unknown'

    def get_setters(self):
        """
        Get the attribute strings with setters for object h
        """

        setters = []
        for name in dir(self.o):
            if not name.startswith('set_'): continue
            o = getattr(self.o,name)
            if not callable(o): continue
            func = o
            if self.is_alias(func): continue
            setters.append(name[4:])
        return setters

    def is_alias(self, o):
        ds = o.__doc__
        if ds is None: return False
        return ds.startswith('alias for ')
    
    def aliased_name(self, s):
        """
        return 'fullname or alias' if s has an alias.  
        """
        if self.aliasd.has_key(s):
            return '%s or %s' % (s, self.aliasd[s])
        else: return s

    def pprint_setters(self, prop=None):
        """
        if prop is None, return a list of strings of all settable properies
        and their valid values

        if prop is not None, it is a valid property name and that
        property will be returned as a string of property : valid
        values
        """
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return '    %s: %s' %(prop, accepts)

        attrs = self.get_setters()
        attrs.sort()
        lines = []

        for prop in attrs:
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)
            lines.append('    %s: %s' %(name, accepts))
        return lines

    def pprint_getters(self):
        """
        return the getters and actual values as list of strings'
        """
        getters = [name for name in dir(self.o)
                   if name.startswith('get_')
                   and callable(getattr(self.o, name))]
        getters.sort()
        lines = []
        for name in getters:
            func = getattr(self.o, name)
            if self.is_alias(func): continue
            try: val = func()
            except: continue
            if hasattr(val, 'shape') and len(val)>6:
                s = str(val[:6]) + '...'
            else:
                s = str(val)                
            name = self.aliased_name(name[4:])
            lines.append('    %s = %s' %(name, s))
        return lines
        
        
def set(h, *args, **kwargs):
    """
matlab(TM) and pylab allow you to use set and get to set and get
object properties, as well as to do introspection on the object
For example, to set the linestyle of a line to be dashed, you can do

  >>> line, = plot([1,2,3])
  >>> set(line, linestyle='--')

If you want to know the valid types of arguments, you can provide the
name of the property you want to set without a value

  >>> set(line, 'linestyle')
      linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]

If you want to see all the properties that can be set, and their
possible values, you can do


  >>> set(line)
      ... long output listing omitted'

set operates on a single instance or a list of instances.  If you are
in quey mode introspecting the possible values, only the first
instance in the sequnce is used.  When actually setting values, all
the instances will be set.  Eg, suppose you have a list of two lines,
the following will make both lines thicker and red

    >>> x = arange(0,1.0,0.01)
    >>> y1 = sin(2*pi*x)
    >>> y2 = sin(4*pi*x)
    >>> lines = plot(x, y1, x, y2)
    >>> set(lines, linewidth=2, color='r')

Set works with the matlab(TM) style string/value pairs or with python
kwargs.  For example, the following are equivalent

    >>> set(lines, 'linewidth', 2, 'color', r')  # matlab style
    >>> set(lines, linewidth=2, color='r')       # python style
    """

    insp = _ObjectInspector(h)

    if len(kwargs)==0 and len(args)==0:
        print '\n'.join(insp.pprint_setters())
        return
    
    if len(kwargs)==0 and len(args)==1:
        print insp.pprint_setters(prop=args[0])
        return
    
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
    """
    
    try:
        a = gcf().add_subplot(*args, **kwargs)        
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        raise RuntimeError, msg
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


    ax2 = gcf().add_axes(ax.get_position(), sharex=ax, frameon=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    draw_if_interactive()
    return ax2


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
    else: raise RuntimeError('Illegal number of arguments to xticks')
    if len(kwargs):
        for l in labels:
            l.update(kwargs)
            
    draw_if_interactive()
    return locs, silent_list('Text xticklabel', labels)



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
    else: raise RuntimeError('Illegal number of arguments to yticks')
    if len(kwargs):
        for l in labels:
            l.update(kwargs)
    
    draw_if_interactive()

    return ( locs, 
             silent_list('Text yticklabel', labels)
             )



def polar(*args, **kwargs):
    """
    POLAR(theta, r)

    Make a polar plot.  Multiple theta, r arguments are supported,
    with format strings, as in plot.
    """
    ax = gca(polar=True)
    return ax.plot(*args, **kwargs)


def over(func, *args, **kwargs):
    """
Call func(*args, **kwargs) with hold(True) and then restore the hold state
    """
    h = ishold()
    hold(True)
    func(*args, **kwargs)
    hold(h)

def ioff():
    'turn interactive mode off'
    matplotlib.interactive(False)

def ion():
    'turn interactive mode on'
    matplotlib.interactive(True)


def switch_backend(newbackend):
    close('all')
    global new_figure_manager, error_msg,  draw_if_interactive, show
    matplotlib.use(newbackend)
    reload(backends)
    from backends import new_figure_manager, error_msg, \
         draw_if_interactive, show

    
### The following functions were autogenerated by the boilerplate.py
### script.  They are simple wrappers around the Axes methods of the
### same name


# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axhline(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axhline(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
axhline.__doc__ = Axes.axhline.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axhspan(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axhspan(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
axhspan.__doc__ = Axes.axhspan.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axvline(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axvline(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
axvline.__doc__ = Axes.axvline.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def axvspan(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().axvspan(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
axvspan.__doc__ = Axes.axvspan.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def bar(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().bar(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
bar.__doc__ = Axes.bar.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def barh(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().barh(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
barh.__doc__ = Axes.barh.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cohere(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().cohere(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
cohere.__doc__ = Axes.cohere.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def contour(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().contour(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        if ret[1].mappable is not None: gci._current = ret[1].mappable
        hold(b)
        return ret
contour.__doc__ = Axes.contour.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def csd(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().csd(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
csd.__doc__ = Axes.csd.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def errorbar(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().errorbar(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
errorbar.__doc__ = Axes.errorbar.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def fill(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().fill(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
fill.__doc__ = Axes.fill.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hist(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().hist(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
hist.__doc__ = Axes.hist.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hlines(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().hlines(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
hlines.__doc__ = Axes.hlines.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def imshow(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().imshow(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        gci._current = ret
        hold(b)
        return ret
imshow.__doc__ = Axes.imshow.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def loglog(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().loglog(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
loglog.__doc__ = Axes.loglog.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pcolor(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().pcolor(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        gci._current = ret
        hold(b)
        return ret
pcolor.__doc__ = Axes.pcolor.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pcolor_classic(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().pcolor_classic(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
pcolor_classic.__doc__ = Axes.pcolor_classic.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pie(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().pie(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
pie.__doc__ = Axes.pie.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def plot(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().plot(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
plot.__doc__ = Axes.plot.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def plot_date(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().plot_date(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
plot_date.__doc__ = Axes.plot_date.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def psd(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().psd(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
psd.__doc__ = Axes.psd.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def scatter(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().scatter(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        gci._current = ret
        hold(b)
        return ret
scatter.__doc__ = Axes.scatter.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def scatter_classic(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().scatter_classic(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
scatter_classic.__doc__ = Axes.scatter_classic.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def semilogx(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().semilogx(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
semilogx.__doc__ = Axes.semilogx.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def semilogy(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().semilogy(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
semilogy.__doc__ = Axes.semilogy.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def specgram(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().specgram(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        gci._current = ret[-1]
        hold(b)
        return ret
specgram.__doc__ = Axes.specgram.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spy(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().spy(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
spy.__doc__ = Axes.spy.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spy2(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().spy2(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        gci._current = ret
        hold(b)
        return ret
spy2.__doc__ = Axes.spy2.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def stem(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().stem(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
stem.__doc__ = Axes.stem.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def vlines(*args, **kwargs):
    # allow callers to override the hold state by passing hold=True|False
    b = ishold()
    h = popd(kwargs, 'hold', None)
    if h is not None:
        hold(h)
    try:
        ret =  gca().vlines(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
        hold(b)
    else:
        draw_if_interactive()
        
        hold(b)
        return ret
vlines.__doc__ = Axes.vlines.__doc__ + """
Addition kwargs: hold = [True|False] overrides default hold state"""

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cla(*args, **kwargs):
    try:
        ret =  gca().cla(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
cla.__doc__ = Axes.cla.__doc__

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def grid(*args, **kwargs):
    try:
        ret =  gca().grid(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
grid.__doc__ = Axes.grid.__doc__

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def legend(*args, **kwargs):
    try:
        ret =  gca().legend(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
legend.__doc__ = Axes.legend.__doc__

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def table(*args, **kwargs):
    try:
        ret =  gca().table(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
table.__doc__ = Axes.table.__doc__

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def text(*args, **kwargs):
    try:
        ret =  gca().text(*args, **kwargs)
    except ValueError, msg:
        msg = raise_msg_to_str(msg)
        error_msg(msg)
    else:
        draw_if_interactive()
        return ret
text.__doc__ = Axes.text.__doc__

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def autumn():
    'set the default colormap to autumn and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='autumn')
    im = gci()
    if im is not None:
        im.set_cmap(cm.autumn)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def bone():
    'set the default colormap to bone and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='bone')
    im = gci()
    if im is not None:
        im.set_cmap(cm.bone)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def cool():
    'set the default colormap to cool and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='cool')
    im = gci()
    if im is not None:
        im.set_cmap(cm.cool)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def copper():
    'set the default colormap to copper and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='copper')
    im = gci()
    if im is not None:
        im.set_cmap(cm.copper)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def flag():
    'set the default colormap to flag and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='flag')
    im = gci()
    if im is not None:
        im.set_cmap(cm.flag)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def gray():
    'set the default colormap to gray and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='gray')
    im = gci()
    if im is not None:
        im.set_cmap(cm.gray)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hot():
    'set the default colormap to hot and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='hot')
    im = gci()
    if im is not None:
        im.set_cmap(cm.hot)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def hsv():
    'set the default colormap to hsv and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='hsv')
    im = gci()
    if im is not None:
        im.set_cmap(cm.hsv)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def jet():
    'set the default colormap to jet and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='jet')
    im = gci()
    if im is not None:
        im.set_cmap(cm.jet)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def pink():
    'set the default colormap to pink and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='pink')
    im = gci()
    if im is not None:
        im.set_cmap(cm.pink)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def prism():
    'set the default colormap to prism and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='prism')
    im = gci()
    if im is not None:
        im.set_cmap(cm.prism)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def spring():
    'set the default colormap to spring and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='spring')
    im = gci()
    if im is not None:
        im.set_cmap(cm.spring)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def summer():
    'set the default colormap to summer and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='summer')
    im = gci()
    if im is not None:
        im.set_cmap(cm.summer)
    draw_if_interactive()

# This function was autogenerated by boilerplate.py.  Do not edit as
# changes will be lost
def winter():
    'set the default colormap to winter and apply to current image if any.  See help(colormaps) for more information'
    rc('image', cmap='winter')
    im = gci()
    if im is not None:
        im.set_cmap(cm.winter)
    draw_if_interactive()



__nxall__ = [

    # numerix
    'array', 'zeros', 'shape', 'rank', 'size', 'fromstring', 'take',
    'put', 'putmask', 'reshape', 'repeat', 'choose', 'searchsorted',
    'asum', 'cumsum', 'product', 'cumproduct', 'alltrue', 'sometrue',
    'allclose', 'arrayrange', 'arange', 'asarray', 'convolve',
    'swapaxes', 'concatenate', 'transpose', 'sort', 'argsort',
    'argmax', 'argmin', 'innerproduct', 'dot', 'outerproduct',
    'resize', 'indices', 'fromfunction', 'diagonal', 'trace', 'ravel',
    'nonzero', 'shape', 'where', 'compress', 'clip', 'zeros', 'ones',
    'identity', 'add', 'logical_or', 'exp', 'subtract', 'logical_xor',
    'log', 'multiply', 'logical_not', 'log10', 'divide', 'maximum',
    'sin', 'minimum', 'sinh', 'conjugate', 'bitwise_and', 'sqrt',
    'power', 'bitwise_or', 'tan', 'absolute', 'bitwise_xor', 'tanh',
    'negative', 'ceil', 'greater', 'fabs', 'greater_equal', 'floor',
    'less', 'arccos', 'arctan2', 'less_equal', 'arcsin', 'fmod',
    'equal', 'arctan', 'hypot', 'not_equal', 'cos', 'around',
    'logical_and', 'cosh', 'arccosh', 'arcsinh', 'arctanh',
    'cross_correlate', 'pi', 'ArrayType', 'matrixmultiply',

    #numerix.mlab
    'rand', 'randn', 'eye', 'tri', 'diag', 'fliplr', 'flipud',
    'rot90', 'tril', 'triu', 'ptp', 'mean', 'msort', 'median', 'std',
    'cumsum', 'prod', 'cumprod', 'trapz', 'diff', 'cov', 'corrcoef',
    'squeeze', 'kaiser', 'blackman', 'bartlett', 'hanning', 'hamming',
    'sinc', 'eig', 'svd', 'angle', 'roots', 'amin', 'amax',

    #matplotlib.mlab
    'linspace', 'window_hanning', 'window_none', 'conv', 'detrend',
    'detrend_mean', 'detrend_none', 'detrend_linear', 'corrcoef',
    'polyfit', 'polyval', 'vander', 'entropy', 'normpdf', 'levypdf',
    'find', 'trapz', 'prepca', 'fix', 'rem', 'norm', 'orth', 'rank',
    'sqrtm', 'prctile', 'center_matrix', 'meshgrid', 'rk4',
    'exp_safe', 'amap', 'sum_flat', 'mean_flat', 'rms_flat', 'l1norm',
    'l2norm', 'norm', 'frange', 'diagonal_matrix', 'base_repr',
    'binary_repr', 'log2', 'ispower2', 'bivariate_normal',
    
    # some misc
    'inverse', 'eigenvectors', 'fft',
    
    # nx types
    'Int8', 'UInt8', 'Int16', 'UInt16', 'Int32', 'UInt32', 'Float32',
    'Float64', 'Complex32', 'Complex64', 'Float', 'Int', 'Complex',
    ]


__plotting_all__ = [
    # plotting
    'plotting', 'colormaps', 'get_current_fig_manager', 'connect',
    'disconnect', 'get_plot_commands', 'raise_msg_to_str', 'axis',
    'axes', 'delaxes', 'clim', 'close', 'clf', 'colorbar', 'draw',
    'figtext', 'figimage', 'figlegend', 'figure', 'gca', 'gcf', 'gci',
    'get', 'hold', 'ishold', 'isinteractive', 'imread', 'load', 'rc',
    'rcdefaults', 'save', 'savefig', 'set', 'subplot', 'twinx', 'title',
    'xlabel', 'ylabel', 'xlim', 'ylim', 'xticks', 'rgrids',
    'thetagrids', 'yticks', 'polar', 'over', 'ioff', 'ion', 'axhline',
    'axhspan', 'axvline', 'axvspan', 'bar', 'barh', 'cohere',
    'contour', 'csd', 'errorbar', 'fill', 'hist', 'hlines', 'imshow',
    'loglog', 'pcolor', 'pcolor_classic', 'pie', 'plot', 'plot_date',
    'psd', 'scatter', 'scatter_classic', 'semilogx', 'semilogy',
    'specgram', 'spy', 'spy2', 'stem', 'vlines', 'cla', 'grid',
    'legend', 'table', 'text', 'autumn', 'bone', 'cool', 'copper',
    'flag', 'gray', 'hot', 'hsv', 'jet', 'pink', 'prism', 'spring',
    'summer', 'winter',

    # classes and modules

    'Line2D', 'Polygon', 'Rectangle', 'Circle', 'Text', 'cm',
    'get_cmap', 'normalize',

    'show', 'rcParams', 'switch_backend'
    ]


__all__ = tuple(list(__dates_all__) + __nxall__ + list(ticker.__all__) + __plotting_all__)
