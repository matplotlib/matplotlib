#!/usr/bin/env python
from pylab import *

def f(t):
    'a damped exponential'
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, .2)


l = plot(t1, f(t1), 'ro')
setp(l, 'markersize', 30)
setp(l, 'markerfacecolor', 'b')

show()

