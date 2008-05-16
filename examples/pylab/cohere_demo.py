#!/usr/bin/env python
"""
Compute the coherence of two signals
"""
import numpy as n

from pylab import figure, show

dt = 0.01
t = n.arange(0, 30, dt)
Nt = len(t)
nse1 = n.random.randn(Nt)                 # white noise 1
nse2 = n.random.randn(Nt)                 # white noise 2
r = n.exp(-t/0.05)

cnse1 = n.convolve(nse1, r)*dt   # colored noise 1
cnse1 = cnse1[:Nt]
cnse2 = n.convolve(nse2, r)*dt   # colored noise 2
cnse2 = cnse2[:Nt]

# two signals with a coherent part and a random part
s1 = 0.01*n.sin(2*n.pi*10*t) + cnse1
s2 = 0.01*n.sin(2*n.pi*10*t) + cnse2

fig = figure()
ax = fig.add_subplot(211)
ax.plot(t, s1, 'b-', t, s2, 'g-')
ax.set_xlim(0,5)
ax.set_xlabel('time')
ax.set_ylabel('s1 and s2')

ax = fig.add_subplot(212)
cxy, f = ax.cohere(s1, s2, 256, 1./dt)

show()


