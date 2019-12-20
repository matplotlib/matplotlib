"""
================
Date tick labels
================

Show how to make date plots in Matplotlib using date tick locators and
formatters.  See :doc:`/gallery/ticks_and_spines/major_minor_demo` for more
information on controlling major and minor ticks.

All Matplotlib date plotting is done by converting date instances into
days since 0001-01-01 00:00:00 UTC plus one day (for historical reasons).
The conversion, tick locating and formatting is done behind the scenes
so this is most transparent to you.  The :mod:`matplotlib.dates` module
provides the converter functions `.date2num` and `.num2date`, which convert
`datetime.datetime` and `numpy.datetime64` objects to and from Matplotlib's
internal representation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load a numpy structured array from yahoo csv data with fields date, open,
# close, volume, adj_close from the mpl-data/example directory.  This array
# stores the date as an np.datetime64 with a day unit ('D') in the 'date'
# column.
with cbook.get_sample_data('goog.npz') as datafile:
    data = np.load(datafile)['price_data']

fig, ax = plt.subplots()
ax.plot('date', 'adj_close', data=data)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(data['date'][0], 'Y')
datemax = np.datetime64(data['date'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()
