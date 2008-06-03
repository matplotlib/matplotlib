#!/usr/bin/env python
from matplotlib.pyplot import *
from numpy import sin, exp,  absolute, pi, arange
from numpy.random import normal

def f(t):
    s1 = sin(2*pi*t)
    e1 = exp(-t)
    return absolute((s1*e1))+.05


t = arange(0.0, 5.0, 0.1)
s = f(t)
nse = normal(0.0, 0.3, t.shape) * s

plot(t, s+nse, 'b^')
vlines(t, [0], s)
xlabel('time (s)')
title('Comparison of model with data')
show()

