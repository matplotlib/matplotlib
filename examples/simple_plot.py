#!/usr/bin/env python
"""
Example: simple line plot.
Show how to make and save a simple line plot with labels, title and grid
"""
from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)

#savefig('simple_plot.png')
savefig('simple_plot')

show()
