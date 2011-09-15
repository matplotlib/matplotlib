"""
A collection of modules for collecting, analyzing and plotting
financial data.   User contributions welcome!

"""
#from __future__ import division
import os, warnings
from urllib2 import urlopen

try:
    from hashlib import md5
except ImportError:
    from md5 import md5 #Deprecated in 2.5
import datetime

import numpy as np

from matplotlib import verbose, get_configdir
from matplotlib.dates import date2num
from matplotlib.cbook import iterable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.lines import Line2D, TICKLEFT, TICKRIGHT
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


configdir = get_configdir()
cachedir = os.path.join(configdir, 'finance.cache')


stock_dt = np.dtype([('date', object),
                     ('year', np.int16),
                     ('month', np.int8),
                     ('day', np.int8),
                     ('d', np.float),     # mpl datenum
                     ('open', np.float),
                     ('close', np.float),
                     ('high', np.float),
                     ('low', np.float),
                     ('volume', np.float),
                     ('aclose', np.float)])


def parse_yahoo_historical(fh, adjusted=True, asobject=False):
    """
    Parse the historical data in file handle fh from yahoo finance.

    *adjusted*
      If True (default) replace open, close, high, and low prices with
      their adjusted values. The adjustment is by a scale factor, S =
      adjusted_close/close. Adjusted prices are actual prices
      multiplied by S.

      Volume is not adjusted as it is already backward split adjusted
      by Yahoo. If you want to compute dollars traded, multiply volume
      by the adjusted close, regardless of whether you choose adjusted
      = True|False.


    *asobject*
      If False (default for compatibility with earlier versions)
      return a list of tuples containing

        d, open, close, high, low, volume

      If None (preferred alternative to False), return
      a 2-D ndarray corresponding to the list of tuples.

      Otherwise return a numpy recarray with

        date, year, month, day, d, open, close, high, low,
        volume, adjusted_close

      where d is a floating poing representation of date,
      as returned by date2num, and date is a python standard
      library datetime.date instance.

      The name of this kwarg is a historical artifact.  Formerly,
      True returned a cbook Bunch
      holding 1-D ndarrays.  The behavior of a numpy recarray is
      very similar to the Bunch.

    """

    lines = fh.readlines()

    results = []

    datefmt = '%Y-%m-%d'

    for line in lines[1:]:

        vals = line.split(',')
        if len(vals)!=7:
            continue      # add warning?
        datestr = vals[0]
        #dt = datetime.date(*time.strptime(datestr, datefmt)[:3])
        # Using strptime doubles the runtime. With the present
        # format, we don't need it.
        dt = datetime.date(*[int(val) for val in datestr.split('-')])
        dnum = date2num(dt)
        open, high, low, close =  [float(val) for val in vals[1:5]]
        volume = float(vals[5])
        aclose = float(vals[6])

        results.append((dt, dt.year, dt.month, dt.day,
                        dnum, open, close, high, low, volume, aclose))
    results.reverse()
    d = np.array(results, dtype=stock_dt)
    if adjusted:
        scale = d['aclose'] / d['close']
        scale[np.isinf(scale)] = np.nan
        d['open'] *= scale
        d['close'] *= scale
        d['high'] *= scale
        d['low'] *= scale

    if not asobject:
        # 2-D sequence; formerly list of tuples, now ndarray
        ret = np.zeros((len(d), 6), dtype=np.float)
        ret[:,0] = d['d']
        ret[:,1] = d['open']
        ret[:,2] = d['close']
        ret[:,3] = d['high']
        ret[:,4] = d['low']
        ret[:,5] = d['volume']
        if asobject is None:
            return ret
        return [tuple(row) for row in ret]

    return d.view(np.recarray)  # Close enough to former Bunch return


def fetch_historical_yahoo(ticker, date1, date2, cachename=None,dividends=False):
    """
    Fetch historical data for ticker between date1 and date2.  date1 and
    date2 are date or datetime instances, or (year, month, day) sequences.

    Ex:
    fh = fetch_historical_yahoo('^GSPC', (2000, 1, 1), (2001, 12, 31))

    cachename is the name of the local file cache.  If None, will
    default to the md5 hash or the url (which incorporates the ticker
    and date range)
    
    set dividends=True to return dividends instead of price data.  With
    this option set, parse functions will not work

    a file handle is returned
    """

    ticker = ticker.upper()


    if iterable(date1):
        d1 = (date1[1]-1, date1[2], date1[0])
    else:
        d1 = (date1.month-1, date1.day, date1.year)
    if iterable(date2):
        d2 = (date2[1]-1, date2[2], date2[0])
    else:
        d2 = (date2.month-1, date2.day, date2.year)


    if dividends:
        g='v'
        verbose.report('Retrieving dividends instead of prices')
    else:
        g='d'

    urlFmt = 'http://table.finance.yahoo.com/table.csv?a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&s=%s&y=0&g=%s&ignore=.csv'


    url =  urlFmt % (d1[0], d1[1], d1[2],
                     d2[0], d2[1], d2[2], ticker, g)


    if cachename is None:
        cachename = os.path.join(cachedir, md5(url).hexdigest())
    if os.path.exists(cachename):
        fh = file(cachename)
        verbose.report('Using cachefile %s for %s'%(cachename, ticker))
    else:
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        urlfh = urlopen(url)

        fh = file(cachename, 'w')
        fh.write(urlfh.read())
        fh.close()
        verbose.report('Saved %s data to cache file %s'%(ticker, cachename))
        fh = file(cachename, 'r')

    return fh


def quotes_historical_yahoo(ticker, date1, date2, asobject=False,
                                        adjusted=True, cachename=None):
    """
    Get historical data for ticker between date1 and date2.  date1 and
    date2 are datetime instances or (year, month, day) sequences.

    See :func:`parse_yahoo_historical` for explanation of output formats
    and the *asobject* and *adjusted* kwargs.

    Ex:
    sp = f.quotes_historical_yahoo('^GSPC', d1, d2,
                                asobject=True, adjusted=True)
    returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]
    [n,bins,patches] = hist(returns, 100)
    mu = mean(returns)
    sigma = std(returns)
    x = normpdf(bins, mu, sigma)
    plot(bins, x, color='red', lw=2)

    cachename is the name of the local file cache.  If None, will
    default to the md5 hash or the url (which incorporates the ticker
    and date range)
    """
    # Maybe enable a warning later as part of a slow transition
    # to using None instead of False.
    #if asobject is False:
    #    warnings.warn("Recommend changing to asobject=None")

    fh = fetch_historical_yahoo(ticker, date1, date2, cachename)

    try:
        ret = parse_yahoo_historical(fh, asobject=asobject,
                                            adjusted=adjusted)
        if len(ret) == 0:
            return None
    except IOError, exc:
        warnings.warn('fh failure\n%s'%(exc.strerror[1]))
        return None

    return ret

def plot_day_summary(ax, quotes, ticksize=3,
                     colorup='k', colordown='r',
                     ):
    """
    quotes is a sequence of (time, open, close, high, low, ...) sequences

    Represent the time, open, close, high, low as a vertical line
    ranging from low to high.  The left tick is the open and the right
    tick is the close.

    time must be in float date format - see date2num

    ax          : an Axes instance to plot to
    ticksize    : open/close tick marker in points
    colorup     : the color of the lines where close >= open
    colordown   : the color of the lines where close <  open
    return value is a list of lines added
    """

    lines = []
    for q in quotes:

        t, open, close, high, low = q[:5]

        if close>=open : color = colorup
        else           : color = colordown

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            antialiased=False,   # no need to antialias vert lines
            )

        oline = Line2D(
            xdata=(t, t), ydata=(open, open),
            color=color,
            antialiased=False,
            marker=TICKLEFT,
            markersize=ticksize,
            )

        cline = Line2D(
            xdata=(t, t), ydata=(close, close),
            color=color,
            antialiased=False,
            markersize=ticksize,
            marker=TICKRIGHT)

        lines.extend((vline, oline, cline))
        ax.add_line(vline)
        ax.add_line(oline)
        ax.add_line(cline)


    ax.autoscale_view()

    return lines


def candlestick(ax, quotes, width=0.2, colorup='k', colordown='r',
                alpha=1.0):

    """

    quotes is a sequence of (time, open, close, high, low, ...) sequences.
    As long as the first 5 elements are these values,
    the record can be as long as you want (eg it may store volume).

    time must be in float days format - see date2num

    Plot the time, open, close, high, low as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    ax          : an Axes instance to plot to
    width       : fraction of a day for the rectangle width
    colorup     : the color of the rectangle where close >= open
    colordown   : the color of the rectangle where close <  open
    alpha       : the rectangle alpha level

    return value is lines, patches where lines is a list of lines
    added and patches is a list of the rectangle patches added

    """

    OFFSET = width/2.0

    lines = []
    patches = []
    for q in quotes:
        t, open, close, high, low = q[:5]

        if close>=open :
            color = colorup
            lower = open
            height = close-open
        else           :
            color = colordown
            lower = close
            height = open-close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color='k',
            linewidth=0.5,
            antialiased=True,
            )

        rect = Rectangle(
            xy    = (t-OFFSET, lower),
            width = width,
            height = height,
            facecolor = color,
            edgecolor = color,
            )
        rect.set_alpha(alpha)


        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


def plot_day_summary2(ax, opens, closes, highs, lows, ticksize=4,
                      colorup='k', colordown='r',
                     ):
    """

    Represent the time, open, close, high, low as a vertical line
    ranging from low to high.  The left tick is the open and the right
    tick is the close.

    ax          : an Axes instance to plot to
    ticksize    : size of open and close ticks in points
    colorup     : the color of the lines where close >= open
    colordown   : the color of the lines where close <  open

    return value is a list of lines added
    """

    # note this code assumes if any value open, close, low, high is
    # missing they all are missing

    rangeSegments = [ ((i, low), (i, high)) for i, low, high in zip(xrange(len(lows)), lows, highs) if low != -1 ]

    # the ticks will be from ticksize to 0 in points at the origin and
    # we'll translate these to the i, close location
    openSegments = [  ((-ticksize, 0), (0, 0)) ]

    # the ticks will be from 0 to ticksize in points at the origin and
    # we'll translate these to the i, close location
    closeSegments = [ ((0, 0), (ticksize, 0)) ]


    offsetsOpen = [ (i, open) for i, open in zip(xrange(len(opens)), opens) if open != -1 ]

    offsetsClose = [ (i, close) for i, close in zip(xrange(len(closes)), closes) if close != -1 ]


    scale = ax.figure.dpi * (1.0/72.0)

    tickTransform = Affine2D().scale(scale, 0.0)

    r,g,b = colorConverter.to_rgb(colorup)
    colorup = r,g,b,1
    r,g,b = colorConverter.to_rgb(colordown)
    colordown = r,g,b,1
    colord = { True : colorup,
               False : colordown,
               }
    colors = [colord[open<close] for open, close in zip(opens, closes) if open!=-1 and close !=-1]

    assert(len(rangeSegments)==len(offsetsOpen))
    assert(len(offsetsOpen)==len(offsetsClose))
    assert(len(offsetsClose)==len(colors))

    useAA = 0,   # use tuple here
    lw = 1,      # and here
    rangeCollection = LineCollection(rangeSegments,
                                     colors       = colors,
                                     linewidths   = lw,
                                     antialiaseds = useAA,
                                     )

    openCollection = LineCollection(openSegments,
                                    colors       = colors,
                                    antialiaseds = useAA,
                                    linewidths   = lw,
                                    offsets      = offsetsOpen,
                                    transOffset  = ax.transData,
                                   )
    openCollection.set_transform(tickTransform)

    closeCollection = LineCollection(closeSegments,
                                     colors       = colors,
                                     antialiaseds = useAA,
                                     linewidths   = lw,
                                     offsets      = offsetsClose,
                                     transOffset  = ax.transData,
                                     )
    closeCollection.set_transform(tickTransform)

    minpy, maxx = (0, len(rangeSegments))
    miny = min([low for low in lows if low !=-1])
    maxy = max([high for high in highs if high != -1])
    corners = (minpy, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(rangeCollection)
    ax.add_collection(openCollection)
    ax.add_collection(closeCollection)
    return rangeCollection, openCollection, closeCollection


def candlestick2(ax, opens, closes, highs, lows, width=4,
                 colorup='k', colordown='r',
                 alpha=0.75,
                ):
    """

    Represent the open, close as a bar line and high low range as a
    vertical line.


    ax          : an Axes instance to plot to
    width       : the bar width in points
    colorup     : the color of the lines where close >= open
    colordown   : the color of the lines where close <  open
    alpha       : bar transparency

    return value is lineCollection, barCollection
    """

    # note this code assumes if any value open, close, low, high is
    # missing they all are missing

    delta = width/2.
    barVerts = [ ( (i-delta, open), (i-delta, close), (i+delta, close), (i+delta, open) ) for i, open, close in zip(xrange(len(opens)), opens, closes) if open != -1 and close!=-1 ]

    rangeSegments = [ ((i, low), (i, high)) for i, low, high in zip(xrange(len(lows)), lows, highs) if low != -1 ]



    r,g,b = colorConverter.to_rgb(colorup)
    colorup = r,g,b,alpha
    r,g,b = colorConverter.to_rgb(colordown)
    colordown = r,g,b,alpha
    colord = { True : colorup,
               False : colordown,
               }
    colors = [colord[open<close] for open, close in zip(opens, closes) if open!=-1 and close !=-1]


    assert(len(barVerts)==len(rangeSegments))

    useAA = 0,  # use tuple here
    lw = 0.5,   # and here
    rangeCollection = LineCollection(rangeSegments,
                                     colors       = ( (0,0,0,1), ),
                                     linewidths   = lw,
                                     antialiaseds = useAA,
                                     )


    barCollection = PolyCollection(barVerts,
                                   facecolors   = colors,
                                   edgecolors   = ( (0,0,0,1), ),
                                   antialiaseds = useAA,
                                   linewidths   = lw,
                                   )

    minx, maxx = 0, len(rangeSegments)
    miny = min([low for low in lows if low !=-1])
    maxy = max([high for high in highs if high != -1])

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(barCollection)
    ax.add_collection(rangeCollection)
    return rangeCollection, barCollection

def volume_overlay(ax, opens, closes, volumes,
                   colorup='k', colordown='r',
                   width=4, alpha=1.0):
    """
    Add a volume overlay to the current axes.  The opens and closes
    are used to determine the color of the bar.  -1 is missing.  If a
    value is missing on one it must be missing on all

    ax          : an Axes instance to plot to
    width       : the bar width in points
    colorup     : the color of the lines where close >= open
    colordown   : the color of the lines where close <  open
    alpha       : bar transparency


    """

    r,g,b = colorConverter.to_rgb(colorup)
    colorup = r,g,b,alpha
    r,g,b = colorConverter.to_rgb(colordown)
    colordown = r,g,b,alpha
    colord = { True : colorup,
               False : colordown,
               }
    colors = [colord[open<close] for open, close in zip(opens, closes) if open!=-1 and close !=-1]

    delta = width/2.
    bars = [ ( (i-delta, 0), (i-delta, v), (i+delta, v), (i+delta, 0)) for i, v in enumerate(volumes) if v != -1 ]

    barCollection = PolyCollection(bars,
                                   facecolors   = colors,
                                   edgecolors   = ( (0,0,0,1), ),
                                   antialiaseds = (0,),
                                   linewidths   = (0.5,),
                                   )

    corners = (0, 0), (len(bars), max(volumes))
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    return barCollection


def volume_overlay2(ax, closes, volumes,
                   colorup='k', colordown='r',
                   width=4, alpha=1.0):
    """
    Add a volume overlay to the current axes.  The closes are used to
    determine the color of the bar.  -1 is missing.  If a value is
    missing on one it must be missing on all

    ax          : an Axes instance to plot to
    width       : the bar width in points
    colorup     : the color of the lines where close >= open
    colordown   : the color of the lines where close <  open
    alpha       : bar transparency

    nb: first point is not displayed - it is used only for choosing the
    right color

    """

    return volume_overlay(ax,closes[:-1],closes[1:],volumes[1:],colorup,colordown,width,alpha)


def volume_overlay3(ax, quotes,
                   colorup='k', colordown='r',
                   width=4, alpha=1.0):
    """
    Add a volume overlay to the current axes.  quotes is a list of (d,
    open, close, high, low, volume) and close-open is used to
    determine the color of the bar

    kwarg
    width       : the bar width in points
    colorup     : the color of the lines where close1 >= close0
    colordown   : the color of the lines where close1 <  close0
    alpha       : bar transparency


    """

    r,g,b = colorConverter.to_rgb(colorup)
    colorup = r,g,b,alpha
    r,g,b = colorConverter.to_rgb(colordown)
    colordown = r,g,b,alpha
    colord = { True : colorup,
               False : colordown,
               }

    dates, opens, closes, highs, lows, volumes = zip(*quotes)
    colors = [colord[close1>=close0] for close0, close1 in zip(closes[:-1], closes[1:]) if close0!=-1 and close1 !=-1]
    colors.insert(0,colord[closes[0]>=opens[0]])

    right = width/2.0
    left = -width/2.0


    bars = [ ( (left, 0), (left, volume), (right, volume), (right, 0)) for d, open, close, high, low, volume in quotes]

    sx = ax.figure.dpi * (1.0/72.0)  # scale for points
    sy = ax.bbox.height / ax.viewLim.height

    barTransform = Affine2D().scale(sx,sy)

    dates = [d for d, open, close, high, low, volume in quotes]
    offsetsBars = [(d, 0) for d in dates]

    useAA = 0,  # use tuple here
    lw = 0.5,   # and here
    barCollection = PolyCollection(bars,
                                   facecolors   = colors,
                                   edgecolors   = ( (0,0,0,1), ),
                                   antialiaseds = useAA,
                                   linewidths   = lw,
                                   offsets      = offsetsBars,
                                   transOffset  = ax.transData,
                                   )
    barCollection.set_transform(barTransform)






    minpy, maxx = (min(dates), max(dates))
    miny = 0
    maxy = max([volume for d, open, close, high, low, volume in quotes])
    corners = (minpy, miny), (maxx, maxy)
    ax.update_datalim(corners)
    #print 'datalim', ax.dataLim.bounds
    #print 'viewlim', ax.viewLim.bounds

    ax.add_collection(barCollection)
    ax.autoscale_view()

    return barCollection

def index_bar(ax, vals,
              facecolor='b', edgecolor='l',
              width=4, alpha=1.0, ):
    """
    Add a bar collection graph with height vals (-1 is missing).

    ax          : an Axes instance to plot to
    width       : the bar width in points
    alpha       : bar transparency


    """

    facecolors = (colorConverter.to_rgba(facecolor, alpha),)
    edgecolors = (colorConverter.to_rgba(edgecolor, alpha),)

    right = width/2.0
    left = -width/2.0


    bars = [ ( (left, 0), (left, v), (right, v), (right, 0)) for v in vals if v != -1 ]

    sx = ax.figure.dpi * (1.0/72.0)  # scale for points
    sy = ax.bbox.height / ax.viewLim.height

    barTransform = Affine2D().scale(sx,sy)

    offsetsBars = [ (i, 0) for i,v in enumerate(vals) if v != -1 ]

    barCollection = PolyCollection(bars,
                                   facecolors   = facecolors,
                                   edgecolors   = edgecolors,
                                   antialiaseds = (0,),
                                   linewidths   = (0.5,),
                                   offsets      = offsetsBars,
                                   transOffset  = ax.transData,
                                   )
    barCollection.set_transform(barTransform)






    minpy, maxx = (0, len(offsetsBars))
    miny = 0
    maxy = max([v for v in vals if v!=-1])
    corners = (minpy, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(barCollection)
    return barCollection
