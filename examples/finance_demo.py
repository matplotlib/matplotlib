#!/usr/bin/env python
from matplotlib.matlab import *
from matplotlib.dates import PyDatetimeConverter, MONDAY
from matplotlib.ticker import  DateFormatter, WeekdayLocator, HourLocator, DayLocator
from matplotlib.finance import quotes_historical_yahoo, candlestick,\
     plot_day_summary

import datetime

# you can specify dates in any format you have a converter for.
# matplotlib will convert everything under the hood to seconds since
# the epoch, but you shouldn't have to deal with this
date1 = datetime.date( 2004, 2, 1 )
date2 = datetime.date( 2004, 4, 12 )

# quotes in Eastern time zone
converter = PyDatetimeConverter() 


mondays = WeekdayLocator(MONDAY)    # major ticks on the mondays
hours   = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # Eg, Jan 12
dayFormatter = DateFormatter('%d')      # Eg, 12

quotes = quotes_historical_yahoo(
    'INTC', date1, date2, converter)

ax = subplot(111)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_major_formatter(weekFormatter)
#ax.xaxis.set_minor_formatter(dayFormatter)

plot_day_summary(ax, quotes, ticksize=3, converter=converter)
#candlestick(ax, quotes, width=0.6, converter=converter)

show()

