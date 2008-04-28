#!/usr/bin/env python
"""
Example: simple line plot with NaNs inserted.
"""
from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
t[41:60] = NaN
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('A sine wave with a gap of NaNs between 0.4 and 0.6')
grid(True)

show()
