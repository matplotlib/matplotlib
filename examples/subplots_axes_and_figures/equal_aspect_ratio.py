"""
.. _subplots_axes_and_figures-equal_aspect_ratio:

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

fig, ax = plt.subplots()
ax.plot(t, s, '-', lw=2)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (mV)')
ax.set_title('About as simple as it gets, folks')
ax.grid(True)

ax.set_aspect('equal', 'datalim')

fig.tight_layout()
plt.show()
