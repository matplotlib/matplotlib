#!/usr/bin/env python
'''
Plot lines with points masked out.

This would typically be used with gappy data, to
break the line at the data gaps.
'''

from pylab import *

x = ma.arange(0, 2*pi, 0.02)
y = ma.sin(x)
y1 = sin(2*x)
y2 = sin(3*x)
ym1 = ma.masked_where(y1 > 0.5, y1)
ym2 = ma.masked_where(y2 < -0.5, y2)

lines = plot(x, y, 'r', x, ym1, 'g', x, ym2, 'bo')
setp(lines[0], linewidth = 4)
setp(lines[1], linewidth = 2)
setp(lines[2], markersize = 10)

legend( ('No mask', 'Masked if > 0.5', 'Masked if < -0.5') ,
        loc = 'upper right')
title('Masked line demo')
show()
