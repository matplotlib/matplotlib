"""
=================
hlines and vlines
=================

This example showcases the functions hlines and vlines.
"""

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as btf
import numpy as np
import numpy.random as rnd


def f(t):
    s1 = np.sin(2 * np.pi * t)
    e1 = np.exp(-t)
    return np.abs(s1 * e1) + .05

t = np.arange(0.0, 5.0, 0.1)
s = f(t)
nse = rnd.normal(0.0, 0.3, t.shape) * s

fig = plt.figure(figsize=(12, 6))
vax = fig.add_subplot(121)
hax = fig.add_subplot(122)

vax.plot(t, s + nse, '^')
vax.vlines(t, [0], s)
vax.vlines([1, 2], 0, 1, transform=btf(vax.transData, vax.transAxes), colors='r')
vax.set_xlabel('time (s)')
vax.set_title('Vertical lines demo')

hax.plot(s + nse, t, '^')
hax.hlines(t, [0], s, lw=2)
hax.set_xlabel('time (s)')
hax.set_title('Horizontal lines demo')

plt.show()
