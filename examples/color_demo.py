#!/usr/bin/env python
"""
matplotlib gives you 3 ways to spcify colors,

    1) as a single letter string, ala matab

    2) as an html style hex string

    3) as an R,G,B tuple, where R,G,B, range from 0-1

See help(colors) for more info.
"""
from matplotlib.matlab import *

subplot(111, axisbg=(0.1843, 0.3098, 0.3098))
#subplot(111, axisbg='#ababab')
t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s, 'y')
xlabel('time (s)', color='r')
ylabel('voltage (mV)', color='k')
title('About as silly as it gets, folks', color='#afeeee')
show()
