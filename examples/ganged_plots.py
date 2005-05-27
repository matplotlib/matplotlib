#!/usr/bin/env python
"""
To create plots that share a common axes (visually) you need to set
the axes locations manually by supplying the appropriate axes
rectangles.  Normally you'll want to turn off the tick labels on all
but one of the axes.

In this example the plots share a common xaxis but you can follow the
same logic to supply a common y axis.
"""
from pylab import *

t = arange(0.0, 2.0, 0.01)

s1 = sin(2*pi*t)
s2 = exp(-t)
s3 = s1*s2

# axes rect in relative 0,1 coords left, bottom, width, height.  Turn
# off xtick labels on all but the lower plot
ax1 = axes([0.1, 0.1,  0.8, 0.25])  # lower 
ax2 = axes([0.1, 0.35, 0.8, 0.25], sharex=ax1)  # middle
setp(ax2.get_xticklabels(), visible=False)
ax3 = axes([0.1, 0.6,  0.8, 0.25], sharex=ax1)  # upper
setp(ax3.get_xticklabels(), visible=False)


ax1.plot(t,s1)
ax2.plot(t,s2)
ax3.plot(t,s3)

show()
