"""
=================
Equal aspect axes
=================

This example shows how to make a plot whose x and y axes have a 1:1 aspect
ratio.
"""
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 1.0 + 0.01, 0.01)
s = np.cos(2 * 2 * np.pi * t)

plt.plot(t, s, '-', lw=2)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)

plt.gca().set_aspect('equal', 'datalim')

plt.show()
