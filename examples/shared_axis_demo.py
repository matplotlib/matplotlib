"""
You can share the x or y axis limits for one axis with another by
passing an axes instance as a sharex or sharey kwarg.

Changing the axis limits on one axes will be reflected automatically
in the other, and vice-versa, so when you navigate with the toolbar
the axes will follow each other on their shared axes.  Ditto for
changes in the axis scaling (eg log vs linear).  However, it is
possible to have differences in tick labeling, eg you can selectively
turn off the tick labels on one axes.
"""
from pylab import *

t = arange(0.01, 5.0, 0.01)
s1 = sin(2*pi*t)
s2 = exp(-t)
s3 = sin(4*pi*t)
ax1 = subplot(311)
plot(t,s1)

## share x only
ax2 = subplot(312, sharex=ax1)
plot(t, s2)

# share x and y
ax3 = subplot(313,  sharex=ax1, sharey=ax1)
plot(t, s3)
xlim(0.01,5.0)
#set(gca(), xscale='log')
show()
