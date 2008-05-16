#!/usr/bin/env python
from pylab import *
t = arange(0.0, 1.01, 0.01)
s = sin(2*2*pi*t)

fill(t, s*exp(-5*t), 'r')
grid(True)
show()
