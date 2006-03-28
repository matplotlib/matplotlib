#!/usr/bin/env python

from pylab import *

x1 = arange(0, 2, 0.01)
y1 = sin(2*pi*x1)
y2 = sin(4*pi*x1) + 2

# reverse x and y2 so the polygon fills in order
x = concatenate( (x1,x1[::-1]) )
y = concatenate( (y1,y2[::-1]) )
                 
p = fill(x, y, facecolor='g', alpha=0.5)
show()
                 
