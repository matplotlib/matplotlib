#!/usr/bin/env python

# override the default reporting of coords

from pylab import *

def millions(x):
    return '$%1.1fM' % (x*1e-6)

x =     rand(20)
y =     1e7*rand(20)

fig, ax = subplots()
ax.fmt_ydata = millions
plot(x, y, 'o')

show()

