#!/usr/bin/env python
# -*- noplot -*-
import matplotlib
matplotlib.use('PS')
from pylab import *

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, .1)
t2 = arange(0.0, 5.0, 0.02)
t3 = arange(0.0, 2.0, 0.01)

figure(1)
subplot(211)
l = plot(t1, f(t1), 'k^')
setp(l, markerfacecolor='k', markeredgecolor='r')
title('A tale of 2 subplots', fontsize=14, fontname='Courier')
ylabel('Signal 1', fontsize=12)
subplot(212)
l = plot(t1, f(t1), 'k>')


ylabel('Signal 2', fontsize=12)
xlabel('time (s)', fontsize=12)

show()

