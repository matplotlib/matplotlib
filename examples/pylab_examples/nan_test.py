#!/usr/bin/env python
"""
Example: simple line plots with NaNs inserted.
"""
from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
t[41:60] = NaN

subplot(2,1,1)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('A sine wave with a gap of NaNs between 0.4 and 0.6')
grid(True)

subplot(2,1,2)
t[0] = NaN
t[-1] = NaN
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('More NaNs at 0.0 and 1.0')
grid(True)

show()
