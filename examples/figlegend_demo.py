#!/usr/bin/env python

from pylab import *
ax1 = axes([0.1, 0.1, 0.4, 0.7])
ax2 = axes([0.55, 0.1, 0.4, 0.7])

x = arange(0.0, 2.0, 0.02)
y1 = sin(2*pi*x)
y2 = exp(-x)
l1, l2 = ax1.plot(x, y1, 'rs-', x, y2, 'go')

y3 = sin(4*pi*x)
y4 = exp(-2*x)
l3, l4 = ax2.plot(x, y3, 'yd-', x, y3, 'k^')

figlegend((l1, l2), ('Line 1', 'Line 2'), 'upper left')
figlegend((l3, l4), ('Line 3', 'Line 4'), 'upper right')
show()
