#!/usr/bin/env python
"""
Example: simple line plot.
Show how to make a plot that has equal aspect ratio
"""
from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)

axes().set_aspect('equal', 'datalim')


show()
