"""
============================================
Set default y-axis tick labels on the right
============================================

Example to show how to change two new properties in rcParams to put yaxis tick
lables on either left or right by default.

These properties can also be set in .matplotlibrc:

ytick.labelright
ytick.labelleft
"""


import matplotlib.pyplot as plt
import numpy as np


x = np.array([x for x in range(10)])

_, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

ax[0].plot(x)
ax[0].yaxis.tick_right()

ax[1].plot(x)
ax[1].yaxis.tick_left()

plt.rcParams['ytick.right'], plt.rcParams['ytick.labelright'] = True, True
plt.rcParams['ytick.left'], plt.rcParams['ytick.labelleft'] = False, False

ax[2].plot(x)

plt.show()
