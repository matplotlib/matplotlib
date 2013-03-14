#!/usr/bin/env python

"""
Small demonstration of the hlines and vlines plots.
"""

from matplotlib import pyplot as plt
from numpy import sin, exp,  absolute, pi, arange
from numpy.random import normal


def f(t):
    s1 = sin(2 * pi * t)
    e1 = exp(-t)
    return absolute((s1 * e1)) + .05


t = arange(0.0, 5.0, 0.1)
s = f(t)
nse = normal(0.0, 0.3, t.shape) * s

fig = plt.figure(figsize=(12, 6))
vax = fig.add_subplot(121)
hax = fig.add_subplot(122)

vax.plot(t, s + nse, 'b^')
vax.vlines(t, [0], s)
vax.set_xlabel('time (s)')
vax.set_title('Vertical lines demo')

hax.plot(s + nse, t, 'b^')
hax.hlines(t, [0], s, lw=2)
hax.set_xlabel('time (s)')
hax.set_title('Horizontal lines demo')

plt.show()
