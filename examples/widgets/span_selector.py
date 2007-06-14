#!/usr/bin/env python
"""
The SpanSelector is a mouse widget to select a xmin/xmax range and plot the
detail view of the selected region in the lower axes
"""
import numpy as npy
from pylab import figure, show
from matplotlib.widgets import SpanSelector

fig = figure(figsize=(8,6))
ax = fig.add_subplot(211, axisbg='#FFFFCC')

x = npy.arange(0.0, 5.0, 0.01)
y = npy.sin(2*npy.pi*x) + 0.5*npy.random.randn(len(x))

ax.plot(x, y, '-')
ax.set_ylim(-2,2)
ax.set_title('Press left mouse button and drag to test')

ax2 = fig.add_subplot(212, axisbg='#FFFFCC')
line2, = ax2.plot(x, y, '-')


def onselect(xmin, xmax):
    indmin, indmax = npy.searchsorted(x, (xmin, xmax))
    indmax = min(len(x)-1, indmax)

    thisx = x[indmin:indmax]
    thisy = y[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw()

# set useblit True on gtkagg for enhanced performance
span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red') )


show()
