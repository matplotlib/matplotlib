#!/usr/bin/env python
"""
The SpanSelector is a mouse widget to select a vmin/vmax
range.  When you left click drag in the axes, a rectangle shows the
selected region.  When you release, the rectangle disappears and a
callback is called with min/max.
"""
import pylab
from matplotlib.widgets import SpanSelector

fig = pylab.figure(figsize=(8,6))
ax = fig.add_subplot(211, axisbg='#FFFFCC')

x,y = 4*(pylab.rand(2,100)-.5)
ax.plot(x,y,'o')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_title('Press left mouse button and drag to test')


def onselect(vmin, vmax):
    print vmin, vmax

# set useblit True on gtkagg for enhanced performance
span = SpanSelector(ax, onselect, 'horizontal', useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red') )

ax2 = fig.add_subplot(212)
ax2.plot([1,2,3])

span2 = SpanSelector(ax2, onselect, 'vertical')

pylab.show()
