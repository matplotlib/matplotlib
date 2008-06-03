#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks
"""
import datetime
from pylab import figure, show
from matplotlib.dates import MONDAY, SATURDAY
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter


date1 = datetime.date( 2002, 1, 5 )
date2 = datetime.date( 2003, 12, 1 )

mondays   = WeekdayLocator(MONDAY)    # every monday
months    = MonthLocator(range(1,13), bymonthday=1)           # every month
monthsFmt = DateFormatter("%b '%y")


quotes = quotes_historical_yahoo('INTC', date1, date2)
if not quotes:
    print 'Found no quotes'
    raise SystemExit

dates = [q[0] for q in quotes]
opens = [q[1] for q in quotes]

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, opens, '-')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.xaxis.set_minor_locator(mondays)
ax.autoscale_view()
#ax.xaxis.grid(False, 'major')
#ax.xaxis.grid(True, 'minor')
ax.grid(True)

fig.autofmt_xdate()

#fig.savefig('date_demo2')
show()
