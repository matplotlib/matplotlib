from __future__ import division, generators

import math, sys

from numerix import MLab, absolute, arange, array, asarray, ones, transpose, \
     log, log10, Float, ravel

import mlab
from artist import Artist
from axis import XTick, YTick, XAxis, YAxis
from cbook import iterable, is_string_like, flatten, enumerate, True, False,\
     allequal
from collections import RegularPolyCollection, PolyCollection
from colors import colorConverter, normalize, Colormap
import cm
from cm import ColormapJet, Grayscale
import _image
from ticker import AutoLocator, LogLocator
from ticker import ScalarFormatter, LogFormatter, LogFormatterExponent, LogFormatterMathtext

from image import AxesImage
from legend import Legend
from lines import Line2D, lineStyles, lineMarkers

from mlab import meshgrid
from matplotlib import rcParams
from patches import Rectangle, Circle, Polygon, bbox_artist
from table import Table
from text import Text, _process_text_args
from transforms import Bbox, Point, Value, Affine
from transforms import  Func, LOG10, IDENTITY
from transforms import get_bbox_transform, unit_bbox
from font_manager import FontProperties

import matplotlib

if matplotlib._havedate:
    from dates import YearLocator, MonthLocator, WeekdayLocator, \
             DayLocator, HourLocator, MinuteLocator, DateFormatter,\
             SEC_PER_MIN, SEC_PER_HOUR, SEC_PER_DAY, SEC_PER_WEEK

                     
def _process_plot_format(fmt):
    """
    Process a matlab style color/line style format string.  Return a
    linestyle, color tuple as a result of the processing.  Default
    values are ('-', 'b').  Example format strings include

    'ko'    : black circles
    '.b'    : blue dots
    'r--'   : red dashed lines

    See Line2D.lineStyles and GraphicsContext.colors for all possible
    styles and color format string.

    """

    colors = {
        'b' : 1,
        'g' : 1,
        'r' : 1,
        'c' : 1,
        'm' : 1,
        'y' : 1,
        'k' : 1,
        'w' : 1,
        }

    
    linestyle = 'None'
    marker = 'None'
    color = rcParams['lines.color']

    # handle the multi char special cases and strip them from the
    # string
    if fmt.find('--')>=0:
        linestyle = '--'
        fmt = fmt.replace('--', '')
    if fmt.find('-.')>=0:
        linestyle = '-.'
        fmt = fmt.replace('-.', '')
    
    chars = [c for c in fmt]

    for c in chars:        
        if lineStyles.has_key(c):
            if linestyle != 'None':
                raise ValueError, 'Illegal format string "%s"; two linestyle symbols' % fmt
            
            linestyle = c
        elif lineMarkers.has_key(c):
            if marker != 'None':
                raise ValueError, 'Illegal format string "%s"; two marker symbols' % fmt
            marker = c
        elif colors.has_key(c):
            color = c
        else:
            err = 'Unrecognized character %c in format string' % c
            raise ValueError, err

    if linestyle == 'None' and marker == 'None':
        linestyle = rcParams['lines.linestyle']
        
    return linestyle, marker, color

class _process_plot_var_args:    
    """

    Process variable length arguments to the plot command, so that
    plot commands like the following are supported

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of x, y, fmt are allowed
    """

    def __init__(self, command='plot'):
        self.command = command
        self._clear_color_cycle()
        
    def _clear_color_cycle(self):
        self.colors = ['b','g','r','c','m','y','k']
        # if the default line color is a color format string, move it up
        # in the que
        try: ind = self.colors.index(rcParams['lines.color'])
        except ValueError:
            self.firstColor = rcParams['lines.color']
        else:
            self.colors[0], self.colors[ind] = self.colors[ind], self.colors[0]
            self.firstColor = self.colors[0]

        self.Ncolors = len(self.colors)

        self.count = 0
        
    def __call__(self, *args, **kwargs):
        
        ret =  self._grab_next_args(*args, **kwargs)
        return ret

    def set_lineprops(self, line, **kwargs):
        assert self.command == 'plot', 'set_lineprops only works with "plot"'
        for key, val in kwargs.items():
            funcName = "set_%s"%key
            if hasattr(line,funcName):
                func = getattr(line,funcName)
                func(val)
        
    def set_patchprops(self, fill_poly, **kwargs):
        assert self.command == 'fill', 'set_patchprops only works with "fill"'
        for key, val in kwargs.items():
            funcName = "set_%s"%key
            if hasattr(fill_poly,funcName):
                func = getattr(fill_poly,funcName)
                func(val)


    def is_filled(self, marker):
        filled = ('o', '^', 'v', '<', '>', 's',
                  'd', 'D', 'h', 'H',
                  'p')
        return marker in filled


    def _plot_1_arg(self, y, **kwargs):
        assert self.command == 'plot', 'fill needs at least 2 arguments'
        if self.count==0:
            color = self.firstColor
        else:
            color = self.colors[int(self.count % self.Ncolors)]

        assert(iterable(y))
        try: N=max(y.shape)
        except AttributeError: N = len(y)
        ret =  Line2D(arange(N), y,
                      color = color,
                      markerfacecolor=color,                
                      )
        self.set_lineprops(ret, **kwargs)
        self.count += 1
        return ret

    def _plot_2_args(self, tup2, **kwargs):
        if is_string_like(tup2[1]):

            assert self.command == 'plot', 'fill needs at least 2 non-string arguments'
            y, fmt = tup2
            assert(iterable(y))
            linestyle, marker, color = _process_plot_format(fmt)

            if self.is_filled(marker): mec = None # use default
            else: mec = color                     # use current color
            try: N=max(y.shape)
            except AttributeError: N = len(y)

            ret =  Line2D(xdata=arange(N), ydata=y,
                          color=color, linestyle=linestyle, marker=marker,
                          markerfacecolor=color,
                          markeredgecolor=mec,                          
                          )
            self.set_lineprops(ret, **kwargs)
            return ret
        else:
            
            x,y = tup2
            #print self.count, self.Ncolors, self.count % self.Ncolors
            assert(iterable(x))
            assert(iterable(y))            
            if self.command == 'plot':
                c = self.colors[self.count % self.Ncolors]
                ret =  Line2D(x, y,
                              color = c,
                              markerfacecolor = c,
                              )
                self.set_lineprops(ret, **kwargs)
                self.count += 1
            elif self.command == 'fill':
                ret = Polygon( zip(x,y), fill=True, )
                self.set_patchprops(ret, **kwargs)

            return ret

    def _plot_3_args(self, tup3, **kwargs):
        if self.command == 'plot':
            x, y, fmt = tup3
            assert(iterable(x))
            assert(iterable(y))            

            linestyle, marker, color = _process_plot_format(fmt)
            if self.is_filled(marker): mec = None # use default
            else: mec = color                     # use current color

            ret = Line2D(x, y, color=color,
                         linestyle=linestyle, marker=marker,
                         markerfacecolor=color,
                         markeredgecolor=mec,
                         )
            self.set_lineprops(ret, **kwargs)
        if self.command == 'fill':
            x, y, facecolor = tup3
            ret = Polygon(zip(x,y),
                          facecolor = facecolor,
                          fill=True, 
                          )
            self.set_patchprops(ret, **kwargs)
        return ret

    def _grab_next_args(self, *args, **kwargs):
        
        remaining = args
        while 1:
            if len(remaining)==0: return
            if len(remaining)==1:
                yield self._plot_1_arg(remaining[0], **kwargs)
                remaining = []
                continue
            if len(remaining)==2:
                yield self._plot_2_args(remaining, **kwargs)
                remaining = []
                continue
            if len(remaining)==3:
                if not is_string_like(remaining[2]):
                    raise ValueError, 'third arg must be a format string'
                yield self._plot_3_args(remaining, **kwargs)
                remaining=[]
                continue
            if is_string_like(remaining[2]):
                yield self._plot_3_args(remaining[:3], **kwargs)
                remaining=remaining[3:]
            else:
                yield self._plot_2_args(remaining[:2], **kwargs)
                remaining=remaining[2:]
            #yield self._plot_2_args(remaining[:2])
            #remaining=args[2:]
        



        
class Axes(Artist):
    """
    Emulate matlab's axes command, creating axes with

       Axes(position=[left, bottom, width, height])

    where all the arguments are fractions in [0,1] which specify the
    fraction of the total figure window.  

    axisbg is the color of the axis background

    """

    def __init__(self, fig, rect,
                 axisbg = None, # defaults to rc axes.facecolor
                 frameon = True):
        Artist.__init__(self)


        if axisbg is None: axisbg = rcParams['axes.facecolor']
        self.set_figure(fig)
        self._position = [Value(val) for val in rect]
        self._axisbg = axisbg
        self._frameon = frameon
        self._xscale = 'linear'
        self._yscale = 'linear'        

        l, b, w, h = self._position

        xmin = fig.bbox.ll().x()
        xmax = fig.bbox.ur().x()
        ymin = fig.bbox.ll().y()
        ymax = fig.bbox.ur().y()
        figw = xmax-xmin
        figh = ymax-ymin
        self.left   =  l*figw
        self.bottom =  b*figh
        self.right  =  (l+w)*figw
        self.top    =  (b+h)*figh


        self.bbox = Bbox( Point(self.left, self.bottom),
                          Point(self.right, self.top ),
                          )
        #these will be updated later as data is added
        self.dataLim = unit_bbox()
        self.viewLim = unit_bbox()

        
        self.transData = get_bbox_transform(self.viewLim, self.bbox)
        self.transAxes = get_bbox_transform(unit_bbox(), self.bbox)

        self._hold = rcParams['axes.hold']
        self._connected = {} # a dict from events to (id, func)    
        self.cla()

        # funcs used to format x and y - fall back on major formatters
        self.fmt_xdata = None  
        self.fmt_ydata = None
        
    def format_xdata(self, x):
        """
        return x string formatted.  This function will use the
        attribute self.fmt_xdata if it is callable, else will fall
        back on the xaxis major formatter
        """
        try: return self.fmt_xdata(x)
        except TypeError:
            func = self.xaxis.get_major_formatter()
            return func(x)

    def format_ydata(self, y):
        """
        return y string formatted.  This function will use the
        attribute self.fmt_ydata if it is callable, else will fall
        back on the yaxis major formatter
        """
        try: return self.fmt_ydata(y)
        except TypeError:
            func = self.yaxis.get_major_formatter()
            return func(y)

        
    def has_data(self):
        return (
            len(self.collections) +            
            len(self.images) +
            len(self.lines) +
            len(self.patches))>0
                

    def _set_artist_props(self, a):
        a.set_figure(self.figure)
        if not a.is_transform_set():
            a.set_transform(self.transData)


    def cla(self):
        """
        Clear the current axes        
        """

        # init these w/ some arbitrary numbers - they'll be updated as
        # data is added to the axes

        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)

        self._get_lines = _process_plot_var_args()
        self._get_patches_for_fill = _process_plot_var_args('fill')

        self._gridOn = rcParams['axes.grid']
        self.lines = []
        self.patches = []
        self.texts = []     # text in axis coords
        self.tables = []
        self.artists = []
        self.images = []
        self.legend = None
        self.collections = []  # collection.Collection instances

        self.images = []

        self.grid(self._gridOn)
        self.title =  Text(
            x=0.5, y=1.02, text='',
            fontproperties=FontProperties(size=rcParams['axes.titlesize']),
            verticalalignment='bottom',
            horizontalalignment='center',
            )
        self.title.set_transform(self.transAxes)

        self._set_artist_props(self.title)
        
        self.axesPatch = Rectangle(
            xy=(0,0), width=1, height=1,
            facecolor=self._axisbg,
            edgecolor=rcParams['axes.edgecolor'],
            )
        self.axesPatch.set_figure(self.figure)
        self.axesPatch.set_transform(self.transAxes)
        self.axesPatch.set_linewidth(rcParams['axes.linewidth'])
        self.axison = True
        
    def add_artist(self, a):
        "Add any artist to the axes"
        self.artists.append(a)
        self._set_artist_props(a)

    def add_collection(self, collection):
        self.collections.append(collection)
        self._set_artist_props(collection)
        collection.set_clip_box(self.bbox)

    def get_images(self):
        return self.images

    def get_xscale(self):
        'return the xaxis scale string: log or linear'
        return self._xscale

    def get_yscale(self):
        'return the yaxis scale string: log or linear'
        return self._yscale

    def update_datalim(self, xys):
        """
        Update the data lim bbox with seq of xy tups
        """
        # if no data is set currently, the bbox will ignore it's
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata        
        self.dataLim.update(xys, not self.has_data())

    def add_line(self, l):
        "Add a line to the list of plot lines"
        self._set_artist_props(l)        
        l.set_clip_box(self.bbox)
        xdata = l.get_xdata()
        ydata = l.get_ydata()
        if l.get_transform() != self.transData:
            xys = self._get_verts_in_data_coords(
                l.get_transform(), zip(xdata, ydata))
            xdata, ydata = zip(*xys)

        corners = ( (min(xdata), min(ydata)), (max(xdata), max(ydata)) )

        self.update_datalim(corners)

        self.lines.append(l)

    def _get_verts_in_data_coords(self, trans, xys):
        if trans == self.transData:
            return xys
        # data is not in axis data units.  We must transform it to
        # display and then back to data to get it in data units
        xys = trans.seq_xy_tups(xys)
        return [ self.transData.inverse_xy_tup(xy) for xy in xys]
        
    def add_patch(self, p):
        "Add a line to the list of plot lines"
        self._set_artist_props(p)
        p.set_clip_box(self.bbox)
        xys = self._get_verts_in_data_coords(
            p.get_transform(), p.get_verts())
        self.update_datalim(xys)

        self.patches.append(p)

    def add_table(self, tab):
        "Add a table instance to the list of axes tables"
        self._set_artist_props(tab)
        self.tables.append(tab)


    def autoscale_view(self):
        # if image data only just use the datalim

        if (len(self.images)>0 and
            len(self.lines)==0 and
            len(self.patches)==0):

            self.set_xlim(self.dataLim.intervalx().get_bounds())            

            self.set_ylim(self.dataLim.intervaly().get_bounds())
            return
        
        locator = self.xaxis.get_major_locator()
        self.set_xlim(locator.autoscale())

        locator = self.yaxis.get_major_locator()
        self.set_ylim(locator.autoscale())

    def bar(self, left, height, width=0.8, bottom=0,
            color='b', yerr=None, xerr=None, ecolor='k', capsize=3
            ):
        """
        BAR(left, height)
        
        Make a bar plot with rectangles at
          left, left+width, 0, height
        left and height are Numeric arrays

        Return value is a list of Rectangle patch instances

        BAR(left, height, width, bottom,
            color, yerr, xerr, capsize, yoff)

        xerr and yerr, if not None, will be used to generate errorbars
        on the bar chart

        color specifies the color of the bar
        ecolor specifies the color of any errorbar

        capsize determines the length in points of the error bar caps

        
        The optional arguments color, width and bottom can be either
        scalars or len(x) sequences

        This enables you to use bar as the basis for stacked bar
        charts, or candlestick plots
        """
        if not self._hold: self.cla()

        # left = asarray(left) - width/2
        left = asarray(left)
        height = asarray(height)

        patches = []


        # if color looks like a color string, and RGB tuple or a
        # scalar, then repeat it by len(x)
        if (is_string_like(color) or
            (iterable(color) and len(color)==3 and len(left)!=3) or
            not iterable(color)):
            color = [color]*len(left)


        if not iterable(bottom):
            bottom = array([bottom]*len(left), Float)
        else:
            bottom = asarray(bottom)
        if not iterable(width):
            width = array([width]*len(left), Float)
        else:
            width = asarray(width)

        N = len(left)
        assert len(bottom)==N, 'bar arg bottom must be len(left)'
        assert len(width)==N, 'bar arg width must be len(left) or scalar'
        assert len(height)==N, 'bar arg height must be len(left) or scalar'
        assert len(color)==N, 'bar arg color must be len(left) or scalar'

        right = left + width
        top = bottom + height
        

        args = zip(left, bottom, width, height, color)
        for l, b, w, h, c in args:            
            if h<0:
                b += h
                h = abs(h)
            r = Rectangle(
                xy=(l, b), width=w, height=h,
                facecolor=c,
                )
            self.add_patch(r)
            patches.append(r)
 

        if xerr is not None or yerr is not None:
            self.errorbar(
                left+0.5*width, bottom+height,
                yerr=yerr, xerr=xerr,
                fmt=None, ecolor=ecolor, capsize=capsize)
        self.autoscale_view()
        return patches



    def barh(self, x, y, height=0.8, left=0,
            color='b', yerr=None, xerr=None, ecolor='k', capsize=3
            ):
        """
        BARH(x, y)
        
        The y values give the heights of the center of the bars.  The
        x values give the length of the bars.

        Return value is a list of Rectangle patch instances

    Optional arguments

          height - the height (thickness)  of the bar

          left  - the x coordinate of the left side of the bar

          color specifies the color of the bar
          
          xerr and yerr, if not None, will be used to generate errorbars
           on the bar chart

          ecolor specifies the color of any errorbar

          capsize determines the length in points of the error bar caps

        
        
        The optional arguments color, height and left can be either
        scalars or len(x) sequences
        """
        if not self._hold: self.cla()

        # left = asarray(left) - width/2
        x = asarray(x)
        y = asarray(y)

        patches = []


        # if color looks like a color string, and RGB tuple or a
        # scalar, then repeat it by len(x)
        if (is_string_like(color) or
            (iterable(color) and len(color)==3 and len(left)!=3) or
            not iterable(color)):
            color = [color]*len(x)


        if not iterable(left):
            left = array([left]*len(x), Float)
        else:
            left = asarray(left)
        if not iterable(height):
            height = array([height]*len(x), Float)
        else:
            height = asarray(height)

        N = len(x)
        assert len(left)==N, 'bar arg left must be len(x)'
        assert len(height)==N, 'bar arg height must be len(x) or scalar'
        assert len(y)==N, 'bar arg y must be len(x) or scalar'
        assert len(color)==N, 'bar arg color must be len(x) or scalar'

        

        width = x
        right = left+x
        bottom = y - height/2.
        top = y + height/2.
        

        args = zip(left, bottom, width, height, color)
        for l, b, w, h, c in args:            
            if h<0:
                b += h
                h = abs(h)
            r = Rectangle(
                xy=(l, b), width=w, height=h,
                facecolor=c,
                )
            self.add_patch(r)
            patches.append(r)
 

        if xerr is not None or yerr is not None:
            self.errorbar(
                right, y,
                yerr=yerr, xerr=xerr,
                fmt=None, ecolor=ecolor, capsize=capsize)
        self.autoscale_view()
        return patches



    def clear(self):
        self.cla()
        
    def cohere(self, x, y, NFFT=256, Fs=2, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=0):
        """
        cohere the coherence between x and y.  Coherence is the normalized
        cross spectral density

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
        if not self._hold: self.cla()
        cxy, freqs = mlab.cohere(x, y, NFFT, Fs, detrend, window, noverlap)

        self.plot(freqs, cxy)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)

        return cxy, freqs

    def csd(self, x, y, NFFT=256, Fs=2, detrend=mlab.detrend_none,
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
        if not self._hold: self.cla()
        pxy, freqs = mlab.csd(x, y, NFFT, Fs, detrend, window, noverlap)
        pxy.shape = len(freqs),
        # pxy is complex

        self.plot(freqs, 10*log10(absolute(pxy)))
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly().get_bounds()

        intv = vmax-vmin
        step = 10*int(log10(intv))
        
        ticks = arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxy, freqs

    def draw(self, renderer, *args, **kwargs):
        "Draw everything (plot lines, axes, labels)"
        
        renderer.open_group('axes')
        self.transData.freeze()  # eval the lazy objects
        self.transAxes.freeze()  # eval the lazy objects
        if self.axison:
            if self._frameon: self.axesPatch.draw(renderer)

        if len(self.images)==1:
            im = self.images[0]
            im.draw(renderer)
        elif len(self.images)>1:
            # make a composite image blending alpha
            # list of (_image.Image, ox, oy)

            if not allequal([im.origin for im in self.images]):
                raise ValueError('Composite images with different origins not supported')
            else:
                origin = self.images[0].origin
            ims = [(im.make_image(renderer),0,0) for im in self.images]

                
            im = _image.from_images(self.bbox.height(), self.bbox.width(), ims)
            im.is_grayscale = False
            l, b, w, h = self.bbox.get_bounds()
            ox = l
            oy = self.figure.bbox.height()-(b+h)
            renderer.draw_image(ox, oy, im, origin, self.bbox)
            

        if self.axison:
            self.xaxis.draw(renderer)
            self.yaxis.draw(renderer)


        for c in self.collections:
            c.draw(renderer)

        for p in self.patches:
            p.draw(renderer)

        for line in self.lines:
            line.draw(renderer)


        for t in self.texts:
            t.draw(renderer)

        self.title.draw(renderer)
        if 0: bbox_artist(self.title, renderer)
        # optional artists
        for a in self.artists:
            a.draw(renderer)


        if self.legend is not None:
            self.legend.draw(renderer)

        for table in self.tables:
            table.draw(renderer)

        self.transData.thaw()  # release the lazy objects
        self.transAxes.thaw()  # release the lazy objects
        renderer.close_group('axes')

    def errorbar(self, x, y, yerr=None, xerr=None,
                 fmt='b-', ecolor=None, capsize=3):
        """
        Plot x versus y with error deltas in yerr and xerr.
        Vertical errorbars are plotted if yerr is not None
        Horizontal errorbars are plotted if xerr is not None

        xerr and yerr may be any of:
            a rank-0, Nx1 Numpy array  - symmetric errorbars +/- value
            an N-element list or tuple - symmetric errorbars +/- value
            a rank-1, Nx2 Numpy array  - asymmetric errorbars -column1/+column2

        Alternatively, x, y, xerr, and yerr can all be scalars, which
        plots a single error bar at x, y.
        
        fmt is the plot format symbol for y.  if fmt is None, just
        plot the errorbars with no line symbols.  This can be useful
        for creating a bar plot with errorbars

        ecolor is a matplotlib color arg which gives the color the
        errobar lines; if None, use the marker color.
        
        Return value is a length 2 tuple.  The first element is a list of
        y symbol lines.  The second element is a list of error bar lines.

        capsize is the size of the error bar caps in points
        """
        if not self._hold: self.cla()
        # make sure all the args are iterable arrays
        if not iterable(x): x = asarray([x])
        else: x = asarray(x)

        if not iterable(y): y = asarray([y])
        else: y = asarray(y)

        if xerr is not None:
            if not iterable(xerr): xerr = asarray([xerr])
            else: xerr = asarray(xerr)

        if yerr is not None:
            if not iterable(yerr): yerr = asarray([yerr])
            else: yerr = asarray(yerr)


        if fmt is not None:
            l0, = self.plot(x,y,fmt)
        else: l0 = None
        caplines = []
        barlines = []

        if ecolor is None and l0 is None:
            ecolor = rcParams['lines.color']
        elif ecolor is None:
            ecolor = l0.get_color()
            
        capargs = {'c':ecolor, 'mfc':ecolor, 'mec':ecolor, 'ms':2*capsize}

        if xerr is not None:
            if len(xerr.shape) == 1:
                left  = x-xerr
                right = x+xerr
            else:
                left  = x-xerr[0]
                right = x+xerr[1]

            barlines.extend( self.hlines(y, x, left) )
            barlines.extend( self.hlines(y, x, right) )            
            caplines.extend(self.plot(left, y, '|', **capargs))
            caplines.extend(self.plot(right, y, '|', **capargs))            

        if yerr is not None:
            if len(yerr.shape) == 1:
                lower = y-yerr
                upper = y+yerr
            else:
                lower = y-yerr[0]
                upper = y+yerr[1]

            barlines.extend( self.vlines(x, y, upper ) )
            barlines.extend( self.vlines(x, y, lower ) )            

            caplines.extend(self.plot(x, lower, '_', **capargs))
            caplines.extend(self.plot(x, upper, '_', **capargs))            

        for l in barlines:
            l.set_color(ecolor)

        self.autoscale_view()

        return (l0, caplines+barlines)

    def fill(self, *args, **kwargs):
        """
        Emulate matlab's fill command.  *args is a variable length
        argument, allowing for multiple x,y pairs with an optional
        color format string.  For example, all of the following are
        legal, assuming a is the Axis instance:
        
          a.fill(x,y)            # plot polygon with vertices at x,y
          a.fill(x,y, 'b' )      # plot polygon with vertices at x,y in blue

        An arbitrary number of x, y, color groups can be specified, as in 
          a.fill(x1, y1, 'g', x2, y2, 'r')  

        Returns a list of patches that were added.
        """
        if not self._hold: self.cla()
        patches = []
        for poly in self._get_patches_for_fill(*args, **kwargs):
            self.add_patch( poly )
            patches.append( poly )
        self.autoscale_view()
        return patches
    
    def get_axis_bgcolor(self):
        'Return the axis background color'
        return self._axisbg

    def get_child_artists(self):
        artists = [self.title, self.axesPatch, self.xaxis, self.yaxis]
        artists.extend(self.lines)
        artists.extend(self.patches)
        artists.extend(self.texts)
        if self.legend is not None:
            artists.append(self.legend)
        return artists
    
    def get_frame(self):
        "Return the axes Rectangle frame"
        return self.axesPatch

    def get_legend(self):
        'Return the Legend instance, or None if no legend is defined'
        return self.legend


    def get_lines(self):
        return self.lines
    
    def get_xaxis(self):
        "Return the XAxis instance"
        return self.xaxis

    def get_xgridlines(self):
        "Get the x grid lines as a list of Line2D instances"
        return self.xaxis.get_gridlines()

    def get_xlim(self):
        "Get the x axis range [xmin, xmax]"
        return self.viewLim.intervalx().get_bounds()


    def get_xticklabels(self):
        "Get the xtick labels as a list of Text instances"
        return self.xaxis.get_ticklabels()

    def get_xticklines(self):
        "Get the xtick lines as a list of Line2D instances"
        return self.xaxis.get_ticklines()
    

    def get_xticks(self):
        "Return the x ticks as a list of locations"
        return self.xaxis.get_ticklocs()

    def get_yaxis(self):
        "Return the YAxis instance"
        return self.yaxis

    def get_ylim(self):
        "Get the y axis range [ymin, ymax]"
        return self.viewLim.intervaly().get_bounds()

    def get_ygridlines(self):
        "Get the y grid lines as a list of Line2D instances"
        return self.yaxis.get_gridlines()

    def get_yticklabels(self):
        "Get the ytick labels as a list of Text instances"
        return self.yaxis.get_ticklabels() 

    def get_yticklines(self):
        "Get the ytick lines as a list of Line2D instances"
        return self.yaxis.get_ticklines()

    def get_yticks(self):
        "Return the y ticks as a list of locations"
        return self.yaxis.get_ticklocs()  

    def grid(self, b):
        "Set the axes grids on or off; b is a boolean"
        self.xaxis.grid(b)
        self.yaxis.grid(b)

    def hist(self, x, bins=10, normed=0, bottom=0):
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

        """
        if not self._hold: self.cla()
        n,bins = mlab.hist(x, bins, normed)
        width = 0.9*(bins[1]-bins[0])
        patches = self.bar(bins, n, width=width, bottom=bottom)
        return n, bins, patches

    def hold(self, b=None):
        """
        Set the hold state.  If hold is None (default), toggle the
        hold state.  Else set the hold state to boolean value b.

        Eg
        hold()      # toggle hold
        hold(True)  # hold is on
        hold(False) # hold is off
        """
        if b is None: self._hold = not self._hold
        else: self._hold = b

    def set_frame_on(self, b):
        """
        Set whether the axes rectangle patch is drawn with boolean b
        """
        self._frameon = b

    def set_image_extent(self, xmin, xmax, ymin, ymax):
        """
        Set the data units of the image.  This is useful if you want to
        plot other things over the image, eg, lines or scatter
        """
        raise SystemExit('set_image_extent deprecated; please pass extent in imshow constructor; see help(imshow)')
        
    def imshow(self, X,
               cmap = None, 
               norm = None, 
               aspect=None,
               interpolation=None,
               alpha=1.0,
               vmin = None,
               vmax = None,
               origin=None,
               extent=None):
        """

IMSHOW(X) - plot image in array X to current axes, resampling to scale
            to axes size

IMSHOW(X, **kwargs) - Use keyword args to control image scaling,
colormapping etc. See below for details


Display the image in array X to current axes.  X must be a
float array

If X is MxN, assume luminance (grayscale)
If X is MxNx3, assume RGB
If X is MxNx4, assume RGBA

A matplotlib.image.AxesImage instance is returned


The following kwargs are allowed: 

  * cmap is a cm colormap instance, eg cm.jet.  If None, default to rc
    image.cmap value

  * aspect is one of: free or preserve.  if None, default to rc
    image.aspect value

  * interpolation is one of: bicubic bilinear blackman100 blackman256
    blackman64 nearest sinc144 sinc256 sinc64 spline16 or spline36.
    If None, default to rc image.interpolation

  * norm is a matplotlib.colors.normalize instance; default is
    normalization().  This scales luminance -> 0-1. 

  * vmin and vmax are used to scale a luminance image to 0-1.  If
    either is None, the min and max of the luminance values will be
    used.  Note if you pass a norm instance, the settings for vmin and
    vmax will be ignored.

  * alpha = 1.0 : the alpha blending value

  * origin is either upper or lower, which indicates where the [0,0]
    index of the array is in the upper left or lower left corner of
    the axes.  If None, default to rc image.origin

  * extent is a data xmin, xmax, ymin, ymax for making image plots
    registered with data plots.  Default is the image dimensions
    in pixels

    """

        if not self._hold: self.cla()

        if norm is not None: assert(isinstance(norm, normalize))
        if cmap is not None: assert(isinstance(cmap, Colormap))        

        im = AxesImage(self, cmap, norm, aspect, interpolation, origin, extent)
        if norm is None:            
            im.set_clim(vmin, vmax)


        im.set_array(X)
        im.set_alpha(alpha)

        xmin, xmax, ymin, ymax = im.get_extent()

        corners = (xmin, ymin), (xmax, ymax)
        self.update_datalim(corners)
        self.set_xlim((xmin, xmax))
        self.set_ylim((ymin, ymax))                
        self.images.append(im)
        
        return im
        
    def in_axes(self, xwin, ywin):
        return self.bbox.contains(xwin, ywin)

    def hlines(self, y, xmin, xmax, fmt='k-'):
        """
        plot horizontal lines at each y from xmin to xmax.  xmin or
        xmax can be scalars or len(x) numpy arrays.  If they are
        scalars, then the respective values are constant, else the
        widths of the lines are determined by xmin and xmax

        Returns a list of line instances that were added
        """
        linestyle, marker, color = _process_plot_format(fmt)
        
        # todo: fix me for y is scalar and xmin and xmax are iterable
        y = asarray(y)
        xmin = asarray(xmin)
        xmax = asarray(xmax)
        
        if len(xmin)==1:
            xmin = xmin*ones(y.shape, y.typecode())
        if len(xmax)==1:
            xmax = xmax*ones(y.shape, y.typecode())

        if len(xmin)!=len(y):
            raise ValueError, 'xmin and y are unequal sized sequences'
        if len(xmax)!=len(y):
            raise ValueError, 'xmax and y are unequal sized sequences'

        lines = []
        for (thisY, thisMin, thisMax) in zip(y,xmin,xmax):            
            line = Line2D(
                [thisMin, thisMax], [thisY, thisY],
                color=color, linestyle=linestyle, marker=marker,
                )
            self.add_line( line )
            lines.append(line)
        return lines


    def legend(self, *args, **kwargs):
        """
        Place a legend on the current axes at location loc.  Labels are a
        sequence of strings and loc can be a string or an integer
        specifying the legend location

        USAGE: 

          Make a legend with existing lines

          >>> legend()

          legend by itself will try and build a legend using the label
          property of the lines.  You can set the label of a line by
          doing plot(x, y, label='my data') or
          line.set_label('my data')
          
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

        The location codes are

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

        """

        loc = kwargs.get('loc', 1)
        if len(args)==0:
            labels = [line.get_label() for line in self.lines]
            lines = self.lines

        elif len(args)==1:
            # LABELS
            labels = args[0]
            lines = [line for line, label in zip(self.lines, labels)]

        elif len(args)==2:
            if is_string_like(args[1]) or isinstance(args[1], int):
                # LABELS, LOC
                labels, loc = args
                lines = [line for line, label in zip(self.lines, labels)]
            else:
                # LINES, LABELS
                lines, labels = args

        elif len(args)==3:
            # LINES, LABELS, LOC
            lines, labels, loc = args
        else:
            raise RuntimeError('Invalid arguments to legend')

        lines = flatten(lines)
        self.legend = Legend(self, lines, labels, loc)
        return self.legend

    def loglog(self, *args, **kwargs):
        """
        Make a loglog plot with log scaling on the a and y axis.  The args
        to semilog x are the same as the args to plot.  See help plot for
        more info

        Optional keyword args supported are any of the kwargs
        supported by plot or set_xscale or set_yscale.  Notable, for
        log scaling:

        basex: base of the x logarithm
        subsx: the location of the minor ticks; None defaults to range(2,basex)
        basey: base of the y logarithm
        subsy: the location of the minor yticks; None defaults to range(2,basey)

        
        """
        if not self._hold: self.cla()
        dx = {'basex': kwargs.get('basex', 10),
              'subsx': kwargs.get('subsx', None),
              }
        dy = {'basey': kwargs.get('basey', 10),
              'subsy': kwargs.get('subsy', None),
              }

        self.set_xscale('log', **dx)
        self.set_yscale('log', **dy)
        l = self.plot(*args, **kwargs)
        return l


    def panx(self, numsteps):
        "Pan the x axis numsteps (plus pan right, minus pan left)"
        self.xaxis.pan(numsteps)
        xmin, xmax = self.viewLim.intervalx().get_bounds()
        for line in self.lines:
            line.set_xclip(xmin, xmax)
        self._send_xlim_event()
        
    def pany(self, numsteps):
        "Pan the x axis numsteps (plus pan up, minus pan down)"
        self.yaxis.pan(numsteps)
        self._send_ylim_event()



    def pcolor(self, *args, **kwargs):
        """\
PCOLOR(C) - make a pseudocolor plot of matrix C

PCOLOR(X, Y, C) - a pseudo color plot of C on the matrices X and Y

PCOLOR(C, **kwargs) - Use keywork args to control colormapping and
                      scaling; see below

Optional keywork args are shown with their defaults below (you must
use kwargs for these):

  * cmap = cm.jet : a cm Colormap instance from matplotlib.cm.
    defaults to cm.jet
         
  * norm = normalize() : matplotlib.colors.normalize is used to scale
    luminance data to 0,1.

  * vmin=None and vmax=None : vmin and vmax are used in conjunction
    with norm to normalize luminance data.  If either are None, the
    min and max of the color array C is used.  If you pass a norm
    instance, vmin and vmax will be None
        
  * shading = 'flat' : or 'faceted'.  If 'faceted', a black grid is
    drawn around each rectangle; if 'flat', edge colors are same as
    face colors

  * alpha=1.0 : the alpha blending value
  
Return value is a matplotlib.collections.PatchCollection
object

Note, the behavior of meshgrid in matlab is a bit counterintuitive for
x and y arrays.  For example,

    x = arange(7)
    y = arange(5)
    X, Y = meshgrid(x,y)

    Z = rand( len(x), len(y))
    pcolor(X, Y, Z)

will fail in matlab and matplotlib.  You will probably be
happy with

    pcolor(X, Y, transpose(Z))

Likewise, for nonsquare Z,

    pcolor(transpose(Z))

will make the x and y axes in the plot agree with the numrows and
numcols of Z
        """
        if not self._hold: self.cla()

        alpha = kwargs.get('alpha', 1.0)
        norm = kwargs.get('norm')
        cmap = kwargs.get('cmap')        
        vmin = kwargs.get('vmin')
        vmax = kwargs.get('vmax')        
        shading = kwargs.get('shading', 'faceted')

        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = meshgrid(arange(numCols+1), arange(numRows+1) )
        elif len(args)==3:
            X, Y, C = args
        else:
            raise RuntimeError('Illegal arguments to pcolor; see help(pcolor)')
        
        Nx, Ny = X.shape
        

        patches = []


        verts =  [ ( (X[i,j], Y[i,j]),     (X[i+1,j], Y[i+1,j]),
                     (X[i+1,j+1], Y[i+1,j+1]), (X[i,j+1], Y[i,j+1]))
                   for i in range(Nx-1)   for j in range(Ny-1)]


        C = array([C[i,j] for i in range(Nx-1)  for j in range(Ny-1)])
        
        if shading == 'faceted':
            edgecolors =  (0,0,0,1), 
        else:
            edgecolors = 'None'


        collection = PolyCollection(
            verts,
            edgecolors   = edgecolors,
            antialiaseds = (0,),
            linewidths   = (0.25,),
            )


        collection.set_alpha(alpha)
        collection.set_array(C)


        if norm is not None: assert(isinstance(norm, normalize))
        if cmap is not None: assert(isinstance(cmap, Colormap))

        collection.set_cmap(cmap)
        collection.set_norm(norm)

        if norm is not None:
            collection.set_clim(vmin, vmax)


        self.grid(0)

        x = ravel(X)
        y = ravel(Y)
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)

        corners = (minx, miny), (maxx, maxy) 
        self.update_datalim( corners)
        self.autoscale_view()

        # add the collection last
        self.add_collection(collection)
        return collection

    def pcolor_classic(self, *args, **kwargs):
        """
        pcolor(C) - make a pseudocolor plot of matrix C

        pcolor(X, Y, C) - a pseudo color plot of C on the matrices X and Y  

        pcolor(C, cmap=cm.jet) - make a pseudocolor plot of matrix C
        using rectangle patches using a colormap jet.  Colormaps are
        avalible in matplotlib.cm.  You must pass this as a kwarg.
        
        pcolor(C, norm=normalize()) - the normalization function used
        to scale your color data to 0-1.  must be passed as a kwarg.

        pcolor(C, alpha=0.5) - set the alpha of the pseudocolor plot.
        Must be used as a kwarg

        Shading:

          The optional keyword arg shading ('flat' or 'faceted') will
          determine whether a black grid is drawn around each pcolor
          square.  Default 'faceteted'
             e.g.,   
             pcolor(C, shading='flat')  
             pcolor(X, Y, C, shading='faceted')

        returns a list of patch objects

        Note, the behavior of meshgrid in matlab is a bit
        counterintuitive for x and y arrays.  For example,

          x = arange(7)
          y = arange(5)
          X, Y = meshgrid(x,y)

          Z = rand( len(x), len(y))
          pcolor(X, Y, Z)

        will fail in matlab and matplotlib.  You will probably be
        happy with

         pcolor(X, Y, transpose(Z))

        Likewise, for nonsquare Z,

         pcolor(transpose(Z))

        will make the x and y axes in the plot agree with the numrows
        and numcols of Z
        """

        if not self._hold: self.cla()
        shading = kwargs.get('shading', 'faceted')

        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = meshgrid(arange(numCols+1), arange(numRows+1) )
        elif len(args)==3:
            X, Y, C = args
        else:
            raise RuntimeError('Illegal arguments to pcolor; see help(pcolor)')
        

        Nx, Ny = X.shape

        cmap = kwargs.get('cmap', cm.get_cmap())
        norm = kwargs.get('norm', normalize())
        if isinstance(norm, normalize) and not norm.scaled():
            norm.autoscale(C)
 

        alpha = kwargs.get('alpha', 1.0)
        nc = norm(C)
        RGBA = cmap(nc, alpha)

        patches = []
        
        for i in range(Nx-1):
            for j in range(Ny-1):
                color = tuple(RGBA[i,j,:3])
                left = X[i,j]
                bottom = Y[i,j]
                width = X[i,j+1]-left
                height = Y[i+1,j]-bottom
                rect = Rectangle(
                    (left, bottom), width, height,
                    )
                rect.set_facecolor(color)
                rect.set_alpha(RGBA[i,j,3])
                if shading == 'faceted':
                    rect.set_linewidth(0.25)
                    rect.set_edgecolor('k')
                else:
                    rect.set_edgecolor(color)
                self.patches.append(rect)
                rect.set_figure(self.figure)
                rect.set_transform(self.transData)
                patches.append(rect)
        self.grid(0)

        minx = MLab.min(MLab.min(X))
        maxx = MLab.max(MLab.max(X))
        miny = MLab.min(MLab.min(Y))
        maxy = MLab.max(MLab.max(Y))

        corners = (minx, miny), (maxx, maxy) 

        
        self.update_datalim( corners)
        self.autoscale_view()
        return patches


    def plot(self, *args, **kwargs):
        """
Emulate matlab's plot command.  *args is a variable length
argument, allowing for multiple x,y pairs with an optional
format string.  For example, all of the following are legal,
assuming a is the Axis instance:
        
    a.plot(x,y)            # plot Numeric arrays y vs x
    a.plot(x,y, 'bo')      # plot Numeric arrays y vs x with blue circles
    a.plot(y)              # plot y using x as index array 0..N-1
    a.plot(y, 'r+')        # ditto with red plusses

An arbitrary number of x, y, fmt groups can be specified, as in 

    a.plot(x1, y1, 'g^', x2, y2, 'g-')  

Return value is a list of lines that were added

The following line styles are supported:

    -     : solid line
    --    : dashed line
    -.    : dash-dot line
    :     : dotted line
    .     : points
    ,     : pixels
    o     : circle symbols
    ^     : triangle up symbols
    v     : triangle down symbols
    <     : triangle left symbols
    >     : triangle right symbols
    s     : square symbols
    +     : plus symbols
    x     : cross symbols
    D     : diamond symbols
    d     : thin diamond symbols
    1     : tripod down symbols
    2     : tripod up symbols
    3     : tripod left symbols
    4     : tripod right symbols
    h     : hexagon symbols
    H     : rotated hexagon symbols
    p     : pentagon symbols
    |     : vertical line symbols
    _     : horizontal line symbols
    steps : use gnuplot style 'steps' # kwarg only

The following color strings are supported

    b  : blue
    g  : green
    r  : red
    c  : cyan
    m  : magenta
    y  : yellow
    k  : black 
    w  : white

Line styles and colors are combined in a single format string

The kwargs that are can be used to set line properties (any property
that has a set_* method).  You can use this to set a line label (for
auto legends), linewidth, anitialising, marker face color, etc.  Here
is an example:

    plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
    plot([1,2,3], [1,4,9], 'rs',  label='line 2')
    axis([0, 4, 0, 10])
    legend()

If you make multiple lines with one plot command, the kwargs apply
to all those lines, eg

    plot(x1, y1, x2, y2, antialising=False)

Neither line will be antialiased.
        """

        if not self._hold: self.cla()
        lines = []
        for line in self._get_lines(*args, **kwargs): 
            self.add_line(line)
            lines.append(line)

        self.autoscale_view()
        return lines

    def plot_date(self, d, y, fmt='bo', tz=None, **kwargs):
        """
        plot_date(d, y, converter, fmt='bo', tz=None, **kwargs)

        d is a sequence of dates represented as float days since
        0001-01-01 UTC and y are the y values at those dates.  fmt is
        a plot format string.  kwargs are passed on to plot.  See plot
        for more information.

        See matplotlib.dates for helper functions date2num, num2date
        and drange for help on creating the required floating point dates

        tz is the timezone - defaults to rc value
        """

        if not matplotlib._havedate:
            raise SystemExit('plot_date: no dates support - dates require python2.3')
        
        if not self._hold: self.cla()

        ret = self.plot(d, y, fmt, **kwargs)

        span  = self.dataLim.intervalx().span()

        if span==0: span = SEC_PER_HOUR
        
        minutes = span*24*60 
        hours  = span*24
        days   = span
        weeks  = span/7.
        months = span/31. # approx
        years  = span/365.

        numticks = 5
        if years>numticks:
            locator = YearLocator(int(years/numticks), tz=tz)  # define
            fmt = '%Y'
        elif months>numticks:
            locator = MonthLocator(tz=tz)            
            fmt = '%b %Y'
        elif weeks>numticks:
            locator = WeekLocator(interval=math.ceil(weeks/numticks), tz=tz)
            fmt = '%a, %b %d'
        elif days>numticks:
            locator = DayLocator(interval=math.ceil(days/numticks), tz=tz)
            fmt = '%b %d'
        elif hours>numticks:
            locator = HourLocator(interval=math.ceil(hours/numticks), tz=tz)
            fmt = '%H:%M\n%b %d'
        elif minutes>numticks:
            locator = MinuteLocator(interval=math.ceil(minutes/numticks), tz=tz)
            fmt = '%H:%M:%S'
        else:
            locator = MinuteLocator(tz=tz)
            fmt = '%H:%M:%S'

        formatter = DateFormatter(fmt, tz=tz)
        self.xaxis.set_major_locator(locator)
        self.xaxis.set_major_formatter(formatter)
        self.autoscale_view()

        return ret
        
    def psd(self, x, NFFT=256, Fs=2, detrend=mlab.detrend_none,
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

        -- noverlap is the length of overlap between adjacent NFFT
           length segments, and is an integer

        Returns the tuple Pxx, freqs

        For plotting, the power is plotted as 10*log10(pxx)) for decibels,
        though pxx itself is returned

        Refs:
          Bendat & Piersol -- Random Data: Analysis and Measurement
            Procedures, John Wiley & Sons (1986)

        """
        if not self._hold: self.cla()
        pxx, freqs = mlab.psd(x, NFFT, Fs, detrend, window, noverlap)
        pxx.shape = len(freqs),

        self.plot(freqs, 10*log10(pxx))
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectrum (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly().get_bounds()
        intv = vmax-vmin
        step = 10*int(log10(intv))
        ticks = arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxx, freqs

    def get_position(self):
        """
        Return the axes position 
        """
        return [val.get() for val in self._position]

    def set_position(self, pos):
        """
        Set the axes position with pos = left, bottom, width, height
        in relative 0,1 coords
        """
        for num,val in zip(pos, self._position):
            val.set(num)

    def stem(self, x, y, linefmt='b-', markerfmt='bo', basefmt='r-'):
        """

        A stem plot plots vertical lines (using linefmt) at each x
        location from the baseline to y, and places a marker there using
        markerfmt.  A horizontal line at 0 is is plotted using basefmt

        return value is markerline, stemlines, baseline

        See
        http://www.mathworks.com/access/helpdesk/help/techdoc/ref/stem.html
        for details and examples/stem_plot.py for a demo.
        """
        if not self._hold: self.cla()
        markerline, = self.plot(x, y, markerfmt)

        stemlines = []
        for thisx, thisy in zip(x, y):
            l, = self.plot([thisx,thisx], [0, thisy], linefmt)
            stemlines.append(l)

        baseline, = self.plot([min(x), max(x)], [0,0], basefmt)
        return markerline, stemlines, baseline
        
        
    def set_axis_off(self):
        self.axison = False

    def set_axis_on(self):
        self.axison = True

    def scatter(self, x, y, s=20, c='b',
                marker = 'o',
                cmap = None,
                norm = None,
                vmin = None,
                vmax = None,
                alpha=1.0):
        """\

SCATTER(x, y) - make a scatter plot of x vs y

SCATTER(x, y, s) - make a scatter plot of x vs y with size in area
                   given by s

SCATTER(x, y, s, c) - make a scatter plot of x vs y with size in area
                      given by s and colors given by c

SCATTER(x, y, s, c, **kwargs) - control colormapping and scaling with
keyword args; see below

Make a scatter plot of x versus y.  s is a size in points^2 a scalar
or an array of the same length as x or y.  c is a color and can be a
single color format string or an length(x) array of intensities which
will be mapped by the matplotlib.colors.colormap instance cmap

The marker can be one of
        
    's' : square
    'o' : circle
    '^' : triangle up
    '>' : triangle right
    'v' : triangle down
    '<' : triangle left
    'd' : diamond
    'p' : pentagram
    'h' : hexagon
    '8' : octagon

s is a size argument in points squared.

        
Other keyword args; the color mapping and normalization arguments will
on be used if c is an array of floats

  * cmap = cm.jet : a cm Colormap instance from matplotlib.cm.
    defaults to rc image.cmap

  * norm = normalize() : matplotlib.colors.normalize is used to
    scale luminance data to 0,1.
    
  * vmin=None and vmax=None : vmin and vmax are used in conjunction
    with norm to normalize luminance data.  If either are None, the
    min and max of the color array C is used.  Note if you pass a norm
    instance, your settings for vmin and vmax will be ignored

  * alpha =1.0 : the alpha value for the patches        
    """
        if not self._hold: self.cla()

        syms =  { # a dict from symbol to (numsides, angle)           
            's' : (4, math.pi/4.0),  # square
            'o' : (20, 0),           # circle
            '^' : (3,0),             # triangle up
            '>' : (3,math.pi/2.0),   # triangle right
            'v' : (3,math.pi),       # triangle down
            '<' : (3,3*math.pi/2.0), # triangle left
            'd' : (4,0),             # diamond
            'p' : (5,0),             # pentagram
            'h' : (6,0),             # hexagon
            '8' : (8,0),             # octagon
            }

        if not syms.has_key(marker):
            raise ValueError('Unknown marker symbol to scatter')


        numsides, rotation = syms[marker]
        if not is_string_like(c) and iterable(c) and len(c)==len(x):
            colors = None
        else:
            colors = ( colorConverter.to_rgba(c, alpha), )

        if not iterable(s):
            scales = (s,)
        else:
            scales = s

        collection = RegularPolyCollection(
            self.figure.dpi,
            numsides, rotation, scales,
            facecolors = colors,
            offsets = zip(x,y),
            transOffset = self.transData,             
            )
        collection.set_alpha(alpha)
        if colors is None:
            if norm is not None: assert(isinstance(norm, normalize))
            if cmap is not None: assert(isinstance(cmap, Colormap))        

            collection.set_array(c)
            collection.set_cmap(cmap)
            collection.set_norm(norm)            
            
            if norm is None:
                collection.set_clim(vmin, vmax)
            


        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)

        w = maxx-minx
        h = maxy-miny

        # the pad is a little hack to deal with the fact that we don't
        # want to transform all the symbols whose scales are in points
        # to data coords to get the exact bounding box for efficiency
        # reasons.  It can be done right if this is deemed important
        padx, pady = 0.05*w, 0.05*h
        corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady) 
        self.update_datalim( corners)
        self.autoscale_view()

        # add the collection last
        self.add_collection(collection)
        return collection


    def scatter_classic(self, x, y, s=None, c='b'):
        """
        Make a scatter plot of x versus y.  s is a size (in data
        coords) and can be either a scalar or an array of the same
        length as x or y.  c is a color and can be a single color
        format string or an length(x) array of intensities which will
        be mapped by the colormap jet.        

        If size is None a default size will be used
        """
        if not self._hold: self.cla()
        if is_string_like(c):
            c = [c]*len(x)
        elif not iterable(c):
            c = [c]*len(x)
        else:
            norm = normalize()
            norm(c)
            c = cm.jet(c)

        if s is None:
            s = [abs(0.015*(max(y)-min(y)))]*len(x)
        elif not iterable(s):
            s = [s]*len(x)
        
        if len(c)!=len(x):
            raise ValueError, 'c and x are not equal lengths'
        if len(s)!=len(x):
            raise ValueError, 's and x are not equal lengths'

        patches = []
        for thisX, thisY, thisS, thisC in zip(x,y,s,c):
            #print thisX, thisY, thisS, thisC
            circ = Circle( (thisX, thisY),
                           radius=thisS,
                           )
            circ.set_facecolor(thisC)
            self.add_patch(circ)
            patches.append(circ)
        self.autoscale_view()
        return patches

    def semilogx(self, *args, **kwargs):
        """
        Make a semilog plot with log scaling on the x axis.  The args
        to semilog x are the same as the args to plot.  See help plot
        for more info.

        Optional keyword args supported are any of the kwargs
        supported by plot or set_xscale.  Notable, for log scaling:

        basex: base of the logarithm
        subsx: the location of the minor ticks; None defaults to range(2,basex)

        """

        d = {'basex': kwargs.get('basex', 10),
             'subsx': kwargs.get('subsx', None),
             }
             
        self.set_xscale('log', **d)
        l = self.plot(*args, **kwargs)
        return l


    def semilogy(self, *args, **kwargs):
        """
        Make a semilog plot with log scaling on the y axis.  The args to
        semilogy are the same as the args to plot.  See help plot for
        more info.

        Optional keyword args supported are any of the kwargs
        supported by plot or set_yscale.  Notable, for log scaling:

        basey: base of the logarithm
        subsy: the location of the minor ticks; None defaults to range(2,basey)
        
        """
        d = {'basey': kwargs.get('basey', 10),
             'subsy': kwargs.get('subsy', None),
             }

        self.set_yscale('log', **d)
        l = self.plot(*args, **kwargs)
        return l


    def set_axis_bgcolor(self, color):
        self._axisbg = color
        self.axesPatch.set_facecolor(color)
                                
    def set_title(self, label, fontdict=None, **kwargs):
        """
        Set the title for the xaxis

        See the text docstring for information of how override and the
        optional args work

        """
        override = {
            'fontsize':rcParams['axes.titlesize'],
            'verticalalignment' : 'bottom',
            'horizontalalignment' : 'left'
            }

        self.title.set_text(label)
        override = _process_text_args({}, fontdict, **kwargs)
        self.title.update_properties(override)
        return self.title


    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        """
        Set the label for the xaxis

        See the text docstring for information of how override and the
        optional args work

        """

        label = self.xaxis.get_label()
        label.set_text(xlabel)
        override = _process_text_args({}, fontdict, **kwargs)
        label.update_properties(override)
        return label

    def _send_xlim_event(self):
        for cid, func in self._connected.get('xlim_changed', []):
            func(self)

    def _send_ylim_event(self):
        for cid, func in self._connected.get('ylim_changed', []):
            func(self)            
            
        
    def set_xlim(self, v, emit=True):
        """
        Set the limits for the xaxis; v = [xmin, xmax]

        if emit is false, do not trigger an event
        """
        self.viewLim.intervalx().set_bounds(*v)
        if emit: self._send_xlim_event()

        
    def set_xscale(self, value, basex = 10, subsx=None):
        """
        Set the xscaling: 'log' or 'linear'

        if value is 'log', the additional kwargs have the following meaning

        basex: base of the logarithm
        subsx: the location of the minor ticks; None defaults to range(2,basex)

        """

        if subsx is None: subsx = range(2, basex)
        assert(value.lower() in ('log', 'linear', ))
        self._xscale = value
        if value == 'log':
            self.xaxis.set_major_locator(LogLocator(basex))
            self.xaxis.set_major_formatter(LogFormatterMathtext(basex))
            self.xaxis.set_minor_locator(LogLocator(basex,subsx))
            self.transData.get_funcx().set_type(LOG10)
        elif value == 'linear':
            self.xaxis.set_major_locator(AutoLocator())
            self.xaxis.set_major_formatter(ScalarFormatter())
            self.transData.get_funcx().set_type( IDENTITY )
        

    def set_xticklabels(self, labels, fontdict=None, **kwargs):
        """
        Set the xtick labels with list of strings labels
        Return a list of axis text instances
        """
        return self.xaxis.set_ticklabels(labels, fontdict, **kwargs)

    def set_xticks(self, ticks):
        "Set the x ticks with list of ticks"
        return self.xaxis.set_ticks(ticks)
        

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        """
        Set the label for the yaxis

        Defaults override is

            override = {
               'verticalalignment'   : 'center',
               'horizontalalignment' : 'right',
               'rotation'='vertical' : }

        See the text doctstring for information of how override and
        the optional args work
        """
        label = self.yaxis.get_label()
        label.set_text(ylabel)
        override = _process_text_args({}, fontdict, **kwargs)
        label.update_properties(override)
        return label

    def set_ylim(self, v, emit=True):
        """
        Set the limits for the xaxis; v = [ymin, ymax]

        if emit is false, do not trigger an event
        """
        self.viewLim.intervaly().set_bounds(*v)
        if emit: self._send_ylim_event()
        
    def set_yscale(self, value, basey=10, subsy=None):
        """
        Set the yscaling: 'log' or 'linear'

        if value is 'log', the additional kwargs have the following meaning

        basey: base of the logarithm

        subsy: the location of the minor ticks; None are the default range(2,basex)

        """

        if subsy is None: subsy = range(2, basey)
        assert(value.lower() in ('log', 'linear', ))
        self._yscale = value
        if value == 'log':
            self.yaxis.set_major_locator(LogLocator(basey))
            self.yaxis.set_major_formatter(LogFormatterMathtext(basey))
            self.yaxis.set_minor_locator(LogLocator(basey,subsy))
            self.transData.get_funcy().set_type(LOG10)
        elif value == 'linear':
            self.yaxis.set_major_locator(AutoLocator())
            self.yaxis.set_major_formatter(ScalarFormatter())
            self.transData.get_funcy().set_type( IDENTITY )

            

    def set_yticklabels(self, labels, fontdict=None, **kwargs):
        """
        Set the ytick labels with list of strings labels.
        Return a list of Text instances
        """
        return self.yaxis.set_ticklabels(labels, fontdict, **kwargs)
        
    def set_yticks(self, ticks):
        "Set the y ticks with list of ticks"
        return self.yaxis.set_ticks(ticks)

    def specgram(self, x, NFFT=256, Fs=2, detrend=mlab.detrend_none,
                 window=mlab.window_hanning, noverlap=128,
                 cmap = None, xextent=None):
        """
        Compute a spectrogram of data in x.  Data are split into NFFT
        length segements and the PSD of each section is computed.  The
        windowing function window is applied to each segment, and the
        amount of overlap of each segment is specified with noverlap

        See help(psd) for information on the other arguments

        cmap is a colormap; if None use default determined by rc
        
        return value is Pxx, freqs, bins, im

        bins are the time points the spectrogram is calculated over
        freqs is an array of frequencies
        Pxx is a len(times) x len(freqs) array of power
        im is a matplotlib image

        xextent is the image extent in the xaxes xextent=xmin, xmax -
        default 0, max(bins), 0, max(freqs) where bins is the
        return value from matplotlib.mlab.specgram
        """
        if not self._hold: self.cla()
        
        Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
             window, noverlap)


        Z = 10*log10(Pxx)
        Z =  mlab.flipud(Z)

        if xextent is None: xextent = 0, max(bins)
        xmin, xmax = xextent
        extent = xmin, xmax, 0, max(freqs)
        im = self.imshow(Z, cmap, extent=extent)

        return Pxx, freqs, bins, im

    def table(self,              
        cellText=None, cellColours=None,
        cellLoc='right', colWidths=None,
        rowLabels=None, rowColours=None, rowLoc='left',
        colLabels=None, colColours=None, colLoc='center',
        loc='bottom', bbox=None):
        """
        Create a table and add it to the axes.  Returns a table
        instance.  For finer grained control over tables, use the
        Table class and add it to the axes with add_table.

        Thanks to John Gill for providing the class and table.
        """

        # Check we have some cellText
        if cellText is None:
            # assume just colours are needed
            rows = len(cellColours)
            cols = len(cellColours[0])
            cellText = [[''] * rows] * cols

        rows = len(cellText)
        cols = len(cellText[0])
        for row in cellText:
            assert len(row) == cols

        if cellColours is not None:
            assert len(cellColours) == rows
            for row in cellColours:
                assert len(row) == cols
        else:
            cellColours = ['w' * cols] * rows

        # Set colwidths if not given
        if colWidths is None:
            colWidths = [1.0/cols] * cols

        # Check row and column labels
        rowLabelWidth = 0
        if rowLabels is None:
            if rowColours is not None:
                rowLabels = [''] * cols
                rowLabelWidth = colWidths[0]
        elif rowColours is None:
            rowColours = 'w' * rows

        if rowLabels is not None:
            assert len(rowLabels) == rows

        offset = 0
        if colLabels is None:
            if colColours is not None:
                colLabels = [''] * rows
                offset = 1
        elif colColours is None:
            colColours = 'w' * cols
            offset = 1

        if rowLabels is not None:
            assert len(rowLabels) == rows

        # Set up cell colours if not given
        if cellColours is None:
            cellColours = ['w' * cols] * rows

        # Now create the table
        table = Table(self, loc, bbox)
        height = table._approx_text_height()

        # Add the cells
        for row in xrange(rows):
            for col in xrange(cols):
                table.add_cell(row+offset, col,
                               width=colWidths[col], height=height,
                               text=cellText[row][col],
                               facecolor=cellColours[row][col],
                               loc=cellLoc)
        # Do column labels
        if colLabels is not None:
            for col in xrange(cols):
                table.add_cell(0, col,
                               width=colWidths[col], height=height,
                               text=colLabels[col], facecolor=colColours[col],
                               loc=colLoc)

        # Do row labels
        if rowLabels is not None:
            for row in xrange(rows):
                table.add_cell(row+offset, -1,
                               width=rowLabelWidth, height=height,
                               text=rowLabels[row], facecolor=rowColours[row],
                               loc=rowLoc)
            if rowLabelWidth == 0:
                table.auto_set_column_width(-1)

        self.add_table(table)
        return table

    
    def text(self, x, y, text, fontdict=None, **kwargs):
        """
        Add text to axis at location x,y (data coords)

        fontdict is a dictionary to override the default text properties.
        If fontdict is None, the default is

        If len(args) the override dictionary will be:

          'verticalalignment'   : 'bottom',
          'horizontalalignment' : 'left'
          'transform'           : self.transData

        **kwargs can in turn be used to override the override, as in

          a.text(x,y,label, fontsize=12)
        
        will have verticalalignment=bottom and
        horizontalalignment=left but will have a fontsize of 12
        
        
        The Text defaults are
            'color'               : 'k',
            'fontproperties'      : see FontProperties
            'horizontalalignment' : 'left'
            'rotation'            : 'horizontal',
            'verticalalignment'   : 'bottom',
            'transform'           : self.transData,

        the default transform specifies that text is in data coords,
        alternatively, you can specify text in axis coords (0,0 lower
        left and 1,1 upper right).  The example below places text in
        the center of the axes

        ax = subplot(111)
        text(0.5, 0.5,'matplotlib', 
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes,
        )
                

        """
        override = {
            'verticalalignment' : 'bottom',
            'horizontalalignment' : 'left',
            #'verticalalignment' : 'top',            
            'transform' : self.transData,
            }

        override = _process_text_args(override, fontdict, **kwargs)
        t = Text(
            x=x, y=y, text=text,
            )
        self._set_artist_props(t)

        t.update_properties(override)
        self.texts.append(t)

        if t.get_clip_on():  t.set_clip_box(self.bbox)
        return t
    


    def vlines(self, x, ymin, ymax, color='k'):
        """
        Plot vertical lines at each x from ymin to ymax.  ymin or ymax
        can be scalars or len(x) numpy arrays.  If they are scalars,
        then the respective values are constant, else the heights of
        the lines are determined by ymin and ymax

        Returns a list of lines that were added
        """
        

        x = asarray(x)
        ymin = asarray(ymin)
        ymax = asarray(ymax)

        if len(ymin)==1:
            ymin = ymin*ones(x.shape, x.typecode())
        if len(ymax)==1:
            ymax = ymax*ones(x.shape, x.typecode())


        if len(ymin)!=len(x):
            raise ValueError, 'ymin and x are unequal sized sequences'
        if len(ymax)!=len(x):
            raise ValueError, 'ymax and x are unequal sized sequences'

        Y = transpose(array([ymin, ymax]))
        lines = []
        for thisX, thisY in zip(x,Y):
            line = Line2D(
                [thisX, thisX], thisY, color=color, linestyle='-',
                )
            self.add_line(line)
            lines.append(line)
        return lines


    def zoomx(self, numsteps):
        """
        Zoom in on the x xaxis numsteps (plus for zoom in, minus for zoom out)
        """
        self.xaxis.zoom(numsteps)
        xmin, xmax = self.viewLim.intervalx().get_bounds()
        for line in self.lines:
            line.set_xclip(xmin, xmax)
        self._send_xlim_event()

    def zoomy(self, numsteps):
        """
        Zoom in on the x xaxis numsteps (plus for zoom in, minus for zoom out)
        """
        self.yaxis.zoom(numsteps)
        self._send_ylim_event()

    _cid = 0
    _events = ('xlim_changed', 'ylim_changed')
    
    def connect(self, s, func):
        """
        Register observers to be notified when certain events occur.
        Register with callback functions with the following signatures.
        The function has the following signature
        
        def func(ax)  # where ax is the instance making the callback.

        The following events can be connected to: %s

        The connection id is is returned - you can use this with
        disconnect to disconnect from the axes event

        """ % ', '.join(Axes._events)

        if s not in Axes._events:
            raise ValueError('You can only connect to the following axes events: %s' % ', '.join(Axes._events))

        cid = Axes._cid
        seq = self._connected.setdefault(s, []).append((cid, func))
        Axes._cid += 1
        return cid

    def disconnect(self, cid):
        'disconnect from the Axes event.'
        for key, val in self._connected.items():
            for item in val:
                if item[0] == cid:
                    self._connected[key].remove(item)
                    return

class Subplot(Axes):
    """
    Emulate matlab's subplot command, creating axes with

      Subplot(numRows, numCols, plotNum)

    where plotNum=1 is the first plot number and increasing plotNums
    fill rows first.  max(plotNum)==numRows*numCols

    You can leave out the commas if numRows<=numCols<=plotNum<10, as
    in

      Subplot(211)    # 2 rows, 1 column, first (upper) plot
    """
    
    def __init__(self, fig, *args, **kwargs):
        # Axes __init__ below

        if len(args)==1:
            s = str(*args)
            if len(s) != 3:
                raise ValueError, 'Argument to subplot must be a 3 digits long'
            rows, cols, num = map(int, s)
        elif len(args)==3:
            rows, cols, num = args
        else:
            raise ValueError, 'Illegal argument to subplot'
        total = rows*cols
        num -= 1    # convert from matlab to python indexing ie num in range(0,total)
        if num >= total:
            raise ValueError, 'Subplot number exceeds total subplots'
        left, right = .125, .9
        bottom, top = .11, .9
        rat = 0.2             # ratio of fig to seperator for multi row/col figs
        totWidth = right-left
        totHeight = top-bottom
    
        figH = totHeight/(rows + rat*(rows-1))
        sepH = rat*figH
    
        figW = totWidth/(cols + rat*(cols-1))
        sepW = rat*figW
    
        rowNum, colNum =  divmod(num, cols)
        
        figBottom = top - (rowNum+1)*figH - rowNum*sepH
        figLeft = left + colNum*(figW + sepW)

        Axes.__init__(self, fig, [figLeft, figBottom, figW, figH], **kwargs)

        self.rowNum = rowNum
        self.colNum = colNum
        self.numRows = rows
        self.numCols = cols

    def is_first_col(self):
        return self.colNum==0

    def is_first_row(self):
        return self.rowNum==0

    def is_last_row(self):
        return self.rowNum==self.numRows-1


    def is_last_col(self):
        return self.colNum==self.numCols-1
