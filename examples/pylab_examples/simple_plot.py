#!/usr/bin/env python
"""
Example: simple line plot.

Show how to make and save a simple line plot with labels, title and
grid

"""
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 1.0+0.01, 0.01)
s = np.cos(2*2*np.pi*t)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)

plt.show()
