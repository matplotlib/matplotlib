#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks

All matplotlib date plotting is done by converting date instances into
seconds since the epoch.  The conversion, tick locating and formatting
is done behind the scenes so this is most transparent to you.  The
dates module provides several converter classes that you can pass to
the date plotting functions which will convert your dates as
necessary.  Currently epoch dates (already converted) are supported
with EpochCOnverter, python2.3 datetime instances are supported with
PyDatetimeConverter, and mx.Datetime is supported with
MxDatetimeConverter.

If you want to define your own converter, the minimum you need to do
is derive a class from dates.DateConverter and implement the epoch and
from_epoch methods.

This example requires an active internet connection since it uses
yahoo finance to get the data for plotting
"""

import sys
try: import datetime
except ImportError:
    print >> sys.stderr, 'This example requires the python2.3 datetime module though you can use the matpltolib date support w/o it'
    sys.exit()

from matplotlib.matlab import *
from matplotlib.dates import PyDatetimeConverter
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.ticker import YearLocator, MonthLocator, DateFormatter

date1 = datetime.date( 1995, 1, 1 )
date2 = datetime.date( 2004, 4, 12 )

pydates = PyDatetimeConverter()

years    = YearLocator(1)   # every year
months   = MonthLocator(1)  # every month
yearsFmt = DateFormatter('%Y')


quotes = quotes_historical_yahoo(
    'INTC', date1, date2, converter=pydates)
if not quotes:
    raise SystemExit

dates = [q[0] for q in quotes]
opens = [q[1] for q in quotes]

ax = subplot(111)
plot_date(dates, opens, pydates)

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()
grid(True)
show()
