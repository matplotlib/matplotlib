import pylab
from matplotlib.widgets import HorizontalSpanSelector

fig = pylab.figure(figsize=(8,6))
ax = fig.add_axes([0.075, 0.25, 0.9, 0.725], axisbg='#FFFFCC')
#ax = fig.add_subplot(111, axisbg='#FFFFCC')
canvas = ax.figure.canvas

x,y = 4*(pylab.rand(2,100)-.5)
ax.plot(x,y,'o')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
title('Press left mouse button and drag to test')


def onselect(xmin, xmax):
    print xmin, xmax

# set useblit True on gtkagg for enhanced performance
span = HorizontalSpanSelector(ax, onselect, useblit=False,
                              rectprops=dict(alpha=0.5, facecolor='red') )

pylab.show()
