"""
=====================================
Clip the data to the axes view limits
=====================================

Demonstrate clipping of line and marker data to the axes view limits. The
``axlim_clip`` keyword argument can be used in any of the 3D plotting
functions.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Generate the random data
np.random.seed(1)
xyz = np.random.rand(25, 3)

# Default behavior is axlim_clip=False
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '-o')

# When axlim_clip=True, note that when a line segment has one vertex outside
# the view limits, the entire line is hidden. The same is true for 3D patches
# if one of their vertices is outside the limits (not shown).
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '--*', axlim_clip=True)

ax.set(xlim=(0.25, 0.75), ylim=(0, 1), zlim=(-1, 1))
ax.legend(['axlim_clip=False (default)', 'axlim_clip=True'])

plt.show()
