#!/usr/bin/env python
import matplotlib.mlab as mlab
from pylab import figure, show
import numpy as npy

x = npy.arange(0, 2, 0.01)
y1 = npy.sin(2*npy.pi*x)
y2 = npy.sin(4*npy.pi*x) + 2

fig = figure()
ax = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)


xs, ys = mlab.poly_between(x, 0, y1)
ax.fill(xs, ys)
ax.set_ylabel('between y1 and 0')

xs, ys = mlab.poly_between(x, y1, 1)
ax2.fill(xs, ys)
ax2.set_ylabel('between y1 and 1')

xs, ys = mlab.poly_between(x, y1, y2)
ax3.fill(xs, ys)
ax3.set_ylabel('between y1 and y2')
ax3.set_xlabel('x')
show()

