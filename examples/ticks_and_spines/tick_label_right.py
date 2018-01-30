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

plt.rcParams['ytick.right'], plt.rcParams['ytick.labelright'] = True, True
plt.rcParams['ytick.left'], plt.rcParams['ytick.labelleft'] = False, False


x = np.array([x for x in range(10)])

_, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

ax[0].plot(x)
ax[0].yaxis.tick_left()

# use default parameter in rcParams, not calling tick_right()
ax[1].plot(x)

plt.show()
