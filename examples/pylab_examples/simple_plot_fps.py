#!/usr/bin/env python

"""
Example: simple line plot.
Show how to make and save a simple line plot with labels, title and grid
"""
# -*- noplot -*-
from __future__ import print_function
from pylab import *

ion()

t = arange(0.0, 1.0+0.001, 0.001)
s = cos(2*2*pi*t)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)

import time

frames = 100.0
t = time.time()
c = time.clock()
for i in range(int(frames)):
    part = i / frames
    axis([0.0, 1.0 - part, -1.0 + part, 1.0 - part])
wallclock = time.time() - t
user = time.clock() - c
print ("wallclock:", wallclock)
print ("user:", user)
print ("fps:", frames / wallclock)
