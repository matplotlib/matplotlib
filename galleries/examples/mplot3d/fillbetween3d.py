"""
=====================
Fill between 3D lines
=====================

Demonstrate how to fill the space between 3D lines with surfaces. Here we
create a sort of "lampshade" shape.
"""

import matplotlib.pyplot as plt
import numpy as np

N = 50
theta = np.linspace(0, 2*np.pi, N)

x1 = np.cos(theta)
y1 = np.sin(theta)
z1 = 0.1 * np.sin(6 * theta)

x2 = 0.6 * np.cos(theta)
y2 = 0.6 * np.sin(theta)
z2 = 2  # Note that scalar values work in addition to length N arrays

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.fill_between(x1, y1, z1, x2, y2, z2, alpha=0.5, edgecolor='k')

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    plot-type: fill_between,
#    level: beginner
