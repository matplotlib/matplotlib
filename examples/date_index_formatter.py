
"""
When plotting daily data, a frequent request is to plot the data
ignoring skips, eg no extra spaces for weekends.  This is particularly
common in financial time series, when you may have data for M-F and
not Sat, Sun and you don't want gaps in the x axis.  The approach is
to simply use the integer index for the xdata and a custom tick
Formatter to get the appropriate date string for a given index.
"""

import numpy
from matplotlib.mlab import csv2rec
from pylab import figure, show
from matplotlib.ticker import Formatter

r = csv2rec('data/msft.csv')[-40:]

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(round(x))
        if ind>=len(self.dates) or ind<0: return ''

        return self.dates[ind].strftime(self.fmt)

formatter = MyFormatter(r.date)

fig = figure()
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(formatter)
ax.plot(numpy.arange(len(r)), r.close, 'o-')
fig.autofmt_xdate()
show()
