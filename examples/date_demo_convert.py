#!/usr/bin/env python

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pylab
PL = pylab
from matplotlib.dates import DayLocator, HourLocator, \
     drange, date2num, timezone
import datetime

matplotlib.rcParams['timezone'] = 'US/Pacific'
tz = timezone('US/Pacific')


date1 = datetime.datetime( 2000, 3, 2, tzinfo=tz)
date2 = datetime.datetime( 2000, 3, 6, tzinfo=tz)
delta = datetime.timedelta(hours=6)
dates = drange(date1, date2, delta)
    
y = PL.arrayrange( len(dates)*1.0)
ysq = y*y

# note new constructor takes days or sequence of days you want to
# tick, not the hour as before.  Default is to tick every day
majorTick = DayLocator(tz=tz)

# the hour locator takes the hour or sequence of hours you want to
# tick, not the base multiple
minorTick = HourLocator(range(0,25,6), tz=tz)
ax = PL.subplot(111)
ax.plot_date(dates, ysq, tz=tz)

# this is superfluous, since the autoscaler should get it right, but
# use date2num and num2date to to convert between dates and floats if
# you want; both date2num and num2date convert an instance or sequence
PL.xlim( dates[0], dates[-1] )
ax.xaxis.set_major_locator(majorTick)
ax.xaxis.set_minor_locator(minorTick)
labels = ax.get_xticklabels()
PL.set(labels,'rotation', 90)
PL.show()
