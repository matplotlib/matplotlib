#!/usr/bin/env python
"""
Set the major ticks on the ints and minor ticks on multiples of 0.2
"""

from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorLocator   = MultipleLocator(1)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(.2)


t = arange(0.0, 10.0, 0.01)
s = sin(2*pi*t)*exp(-t*0.01)

ax = subplot(111)
plot(t,s)

ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)

#for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)

show()
