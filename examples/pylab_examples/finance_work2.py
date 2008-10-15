"""
You need a additional files to run this example.  Save the following
in the same dir as this file

  http://matplotlib.sourceforge.net/screenshots/helpers.py

  http://matplotlib.sourceforge.net/screenshots/msft_nasdaq_d.csv

  http://matplotlib.sourceforge.net/screenshots/__init__.py

"""

import time, os, sys, datetime

from matplotlib import rcParams
from matplotlib.ticker import  IndexLocator, FuncFormatter, NullFormatter, MultipleLocator
from matplotlib.dates import IndexDateFormatter, date2num
from matplotlib.finance import candlestick2, plot_day_summary2, \
     volume_overlay, index_bar
from pylab import *

rcParams['timezone'] = 'US/Eastern'
rc('grid', color='0.75', linestyle='-', linewidth=0.5)

def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most recent (index
    -1) n is an integer

    returns a numeric array of the exponential moving average
    """
    s = array(s)
    ema = []
    j = 1
    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)
    #EMA(current) = ( (Price(current) - EMA(prev) ) xMultiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)
    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return ema

def movavg(s, n):
    """
    returns an n period moving average for the time series s

    s is a list ordered from oldest (index 0) to most recent (index -1)
    n is an integer

        returns a numeric array of the moving average

    See also ema in this module for the exponential moving average.
    """
    s = array(s)
    c = cumsum(s)
    return (c[n-1:] - c[:-n+1]) / float(n-1)

def fill_over(ax, x, y, val, color, over=True):
    """
    Plot filled x,y for all y over val
    if over = False, fill all areas < val
    """
    ybase = asarray(y)-val
    crossings = nonzero(less(ybase[:-1] * ybase[1:],0))

    if ybase[0]>=0: fillon = over
    else:           fillon = not over


    indLast = 0
    for ind in crossings:
        if fillon:
            thisX = x[indLast:ind+1]
            thisY = y[indLast:ind+1]
            thisY[0] = val
            thisY[-1] = val
            ax.fill(thisX, thisY, facecolor=color)
        fillon = not fillon
        indLast = ind


def random_signal(N, tau):
    'generate a length N  random signal with time constant tau'
    t = arange(float(N))
    filter = exp(-t/tau)
    return convolve( randn(N), filter, mode=2)[:len(t)]


# load a numpy record array from yahoo csv data with fields date,
# open, close, volume, adj_close from the mpl-data/example directory.
# The record array stores python datetime.date as an object array in
# the date column
datafile = matplotlib.get_example_data('goog.npy')
r = np.load(datafile).view(np.recarray)
r = r[-250:]

N = len(r)

vind = np.arange(N)

figBG   = 'w'        # the figure background color
axesBG  = '#f6f6f6'  # the axies background color
textsize = 8        # size for axes text

# the demo data are intc from (2003, 9, 1) to (2004, 4, 12 ) with
# dates as epoch; I saved these to a file for ease of debugginh
ticker = 'MSFT'


figure(1, facecolor=figBG)

def get_locator():
    """
    the axes cannot share the same locator, so this is a helper
    function to generate locators that have identical functionality
    """

    return IndexLocator(10, 1)


formatter =  IndexDateFormatter(date2num(r.date), '%b %d %y')

nullfmt   = NullFormatter()         # no labels

def fmt_vol(x,pos):
    if pos>3: return ''  # only label the first 3 ticks
    return '%dM' % int(x*1e-6)

volumeFmt = FuncFormatter(fmt_vol)

left, width = 0.1, 0.8
rect1 = [left, 0.7, width, 0.2]
rect2 = [left, 0.3, width, 0.4]
rect3 = [left, 0.1, width, 0.2]
axUpper      = axes(rect1, axisbg=axesBG)  #left, bottom, width, height
axMiddle     = axes(rect2, axisbg=axesBG, sharex=axUpper)
axMiddleVol  = axMiddle.twinx()
axLower      = axes(rect3, axisbg=axesBG, sharex=axUpper)


axUpper.xaxis.set_major_locator( get_locator() )
axUpper.xaxis.set_major_formatter(nullfmt)
axUpper.grid(True)

# set up two scales on middle axes with left and right ticks
axMiddle.yaxis.tick_left()
axMiddle.xaxis.set_major_formatter(nullfmt)

axMiddleVol.yaxis.set_major_formatter(volumeFmt)
axMiddle.grid(True)

axLower.xaxis.set_major_locator( get_locator() )
axLower.xaxis.set_major_formatter( formatter )
axLower.grid(True)

if 1: ############### Upper axes #################

    # make up a pseudo signal
    purple = '#660033'
    s = random_signal(N, tau=20)
    thresh = 4
    axUpper.plot(s, color=purple)
    # upper horiz line



    axUpper.plot( (0, N), [thresh, thresh], color=purple, linewidth=1)
    # lower horiz line
    axUpper.plot( (0, N), [-thresh, -thresh], color=purple, linewidth=1)


    # fill above threshold
    fill_over(axUpper, vind, s,  thresh,  purple, over=True)
    fill_over(axUpper, vind, s, -thresh,  purple,  over=False)

    t = axUpper.set_title('Google (GOOG)',  fontsize=12)
    t.set_y(1.05)  # move it up a bit higher than the default
    t.set_x(0)  # align the title left, axes coords
    t.set_horizontalalignment('left')  # align the title left, axes coords
    axUpper.yaxis.set_major_locator( MultipleLocator(5) )



    # now add some text
    left, height, top = 0.025, 0.06, 0.85
    t = axUpper.text(left, top, 'RSI(14) 51.0', fontsize=textsize,
                     transform=axUpper.transAxes)


if 1:  ############### Middle axes #################


    candlestick2(axMiddle, r.open, r.close, r.high, r.low, width=0.9)

    # specify the text in axes (0,1) coords.  0,0 is lower left and 1,1 is
    # upper right

    left, height, top = 0.025, 0.06, 0.9
    t1 = axMiddle.text(left, top, '%s daily'%ticker, fontsize=textsize,
                       transform=axMiddle.transAxes)
    t2 = axMiddle.text(left, top-height, 'MA(5)', color='b', fontsize=textsize,
                       transform=axMiddle.transAxes)
    t3 = axMiddle.text(left, top-2*height, 'MA(20)', color='r', fontsize=textsize,
                       transform=axMiddle.transAxes)

    s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' %(
        time.strftime('%d-%b-%Y'),
        r.open[-1], r.high[-1],
        r.low[-1], r.close[-1],
        r.volume[-1]*1e-6,
        r.close[-1]-r.open[-1])
    t4 = axMiddle.text(0.4, top, s, fontsize=textsize,
                       transform=axMiddle.transAxes)


    # now do the moviing average.  I'll use a convolution to simulate a
    # real moving average
    ma5  = movavg(r.adj_close, 5)
    ma20 = movavg(r.adj_close, 20)
    axMiddle.plot(vind[5-1:], ma5,   'b', linewidth=1)
    axMiddle.plot(vind[20-1:], ma20, 'r', linewidth=1)

    axMiddle.set_ylim((300, 800))
    axMiddle.set_yticks(np.arange(800, 800, 100))

    # Now do the volume overlay

    # todo - this is broken
    bars = volume_overlay(axMiddleVol, r.open, r.close, r.volume, alpha=0.5)
    #axMiddleVol.set_ylim(0, 3*r.volume.max())  # use only a third of the viewlim


if 1:  ############### Lower axes #################

    # make up two signals; I don't know what the signals are in real life
    # so I'll just illustrate the plotting stuff
    s1 = random_signal(N, 10)
    s2 = random_signal(N, 20)

    axLower.plot(vind, s1, color=purple)
    axLower.plot(vind, s2, color='k', linewidth=1.0)
    s3 = s2-s1
    axLower.plot(vind, s3, color='#cccc99')  # wheat
    bars = index_bar(axLower, s3, width=2, alpha=0.5,
                     facecolor='#3087c7', edgecolor='#cccc99')
    axLower.yaxis.set_major_locator(MultipleLocator(5))


    # now add some text
    left, height, top = 0.025, 0.06, 0.85

    t = axLower.text(left, top, 'MACD(12,26,9) -0.26', fontsize=textsize,
                     transform=axLower.transAxes)

    # make sure everyone has the same axes limits

    setp(axLower.get_xticklabels(), 'rotation', 45,
        'horizontalalignment', 'right', fontsize=8)

# force all the axes to have the same x data limits
allAxes = (axUpper, axMiddle, axMiddleVol, axLower)
xlim = 0, N
for a in allAxes:
    a.set_xlim(xlim)

for ax in axUpper, axMiddle, axMiddleVol:
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_visible(False)

show()
