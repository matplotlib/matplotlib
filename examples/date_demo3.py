#!/usr/bin/env python
"""
intdate is a class derived from int that is seconds since the epoch
and also provides some helper functions for getting day, month, year,
etc...  You can use intdate with any matplotlib plotting function, not
just plot_date
"""
from datetime import datetime
import time
from matplotlib.dates import intdate
from matplotlib.ticker import MinuteLocator, DateFormatter
from matplotlib.matlab import *

# simulate collecting data every minute starting at midnight
t0 = time.mktime(datetime(2004,04,27).timetuple())
t = t0+arange(0, 2*3600, 60)  # 2 hours sampled every 2 minute
s = rand(len(t))

ax = subplot(111)
ax.xaxis.set_major_locator( MinuteLocator(20) )
ax.xaxis.set_major_formatter( DateFormatter('%H:%M') )
bar(t, s, width=60)
show()



