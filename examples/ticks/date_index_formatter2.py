"""
====================
Date Index Formatter
====================

When plotting daily data, a frequent request is to plot the data
ignoring skips, e.g., no extra spaces for weekends.  This is particularly
common in financial time series, when you may have data for M-F and
not Sat, Sun and you don't want gaps in the x axis.  The approach is
to simply use the integer index for the xdata and a custom tick
Formatter to get the appropriate date string for a given index.
"""

import dateutil.parser
from matplotlib import cbook, dates
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import numpy as np


datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading %s' % datafile)
msft_data = np.genfromtxt(
    datafile, delimiter=',', names=True,
    converters={0: lambda s: dates.date2num(dateutil.parser.parse(s))})


class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos."""
        ind = int(round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return dates.num2date(self.dates[ind]).strftime(self.fmt)


fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(MyFormatter(msft_data['Date']))
ax.plot(msft_data['Close'], 'o-')
fig.autofmt_xdate()
plt.show()
