"""
The HorizontalSpanSelector is a mouse widget to select a xmin/xmax
range.  When you left click drag in the axes, a rectangle shows the
selected region.  When you release, the rectangle disappears and a
callback is called with min/max.
"""
import pylab
from matplotlib.widgets import HorizontalSpanSelector

fig = pylab.figure(figsize=(8,6))
ax = fig.add_subplot(111, axisbg='#FFFFCC')

x,y = 4*(pylab.rand(2,100)-.5)
ax.plot(x,y,'o')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_title('Press left mouse button and drag to test')


def onselect(xmin, xmax):
    print xmin, xmax

# set useblit True on gtkagg for enhanced performance
span = HorizontalSpanSelector(ax, onselect, useblit=False,
                              rectprops=dict(alpha=0.5, facecolor='red') )

pylab.show()
