#!/usr/bin/env python
"""
Compute the cross spectral density of two signals
"""
from __future__ import division
from pylab import *

dt = 0.01
t = arange(0, 30, dt)
nse1 = randn(len(t))                 # white noise 1
nse2 = randn(len(t))                 # white noise 2
r = exp(divide(-t,0.05))

cnse1 = convolve(nse1, r, mode=2)*dt   # colored noise 1
cnse1 = cnse1[:len(t)]
cnse2 = convolve(nse2, r, mode=2)*dt   # colored noise 2
cnse2 = cnse2[:len(t)]

# two signals with a coherent part and a random part
s1 = 0.01*sin(2*pi*10*t) + cnse1
s2 = 0.01*sin(2*pi*10*t) + cnse2

subplot(211)
plot(t, s1, 'b-', t, s2, 'g-')
xlim(0,5)
xlabel('time')
ylabel('s1 and s2')

subplot(212)
cxy, f = csd(s1, s2, 256, 1/dt)
show()


