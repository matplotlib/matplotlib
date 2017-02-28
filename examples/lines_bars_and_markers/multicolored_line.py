'''
==================
Multicolored lines
==================

This example shows how to make a multi-colored line. In this example, the line
is colored based on its derivative.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)

fig, ax = plt.subplots()
line = ax.add_collection(lc)
fig.colorbar(line)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.1, 1.1)

plt.show()
