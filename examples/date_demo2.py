#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks
"""

import sys
try: import datetime
except ImportError:
    print >> sys.stderr, 'This example requires the python2.3 datetime module though you can use the matpltolib date support w/o it'
    sys.exit()
    
from matplotlib.matlab import *
from matplotlib.dates import PyDatetimeConverter, MONDAY, SATURDAY
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.ticker import MonthLocator, WeekdayLocator, DateFormatter


date1 = datetime.date( 2003, 1, 1 )
date2 = datetime.date( 2004, 4, 12 )

pydates = PyDatetimeConverter()

mondays   = WeekdayLocator(MONDAY)  # every monday
months    = MonthLocator(1)           # every month
monthsFmt  = DateFormatter('%b %d')


quotes = quotes_historical_yahoo(
    'INTC', date1, date2, converter=pydates)
if not quotes:
    raise SystemExit

dates = [q[0] for q in quotes]
opens = [q[1] for q in quotes]

ax = subplot(111)
plot_date(dates, opens, pydates)

ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.xaxis.set_minor_locator(mondays)
ax.autoscale_view()
#ax.xaxis.grid(False, 'major')
#ax.xaxis.grid(True, 'minor')

labels = ax.get_xticklabels()
set(labels, 'rotation', 45)

grid(True)
show()
