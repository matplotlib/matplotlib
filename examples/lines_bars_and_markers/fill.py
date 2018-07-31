"""
==============
Fill plot demo
==============

Demo fill plot.
"""

###############################################################################
# First, the most basic fill plot a user can make with matplotlib:
import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 1]
y = [1, 2, 1, 0]

fig, ax = plt.subplots()
ax.fill(x, y)
plt.show()

###############################################################################
# Next, a few more optional features:
#
# * Multiple curves with a single command.
# * Setting the fill color.
# * Setting the opacity (alpha value).


x = np.linspace(0, 1.5 * np.pi, 500)
y1 = np.sin(x)
y2 = np.sin(3 * x)

fig, ax = plt.subplots()

ax.fill(x, y1, 'b', x, y2, 'r', alpha=0.3)

# Outline of the region we've filled in
ax.plot(x, y1, c='b', alpha=0.8)
ax.plot(x, y2, c='r', alpha=0.8)
ax.plot([x[0], x[-1]], [y1[0], y1[-1]], c='b', alpha=0.8)
ax.plot([x[0], x[-1]], [y2[0], y2[-1]], c='r', alpha=0.8)

plt.show()
