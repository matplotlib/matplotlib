'''
==========================================
3D voxel / volumetric plot with rgb colors
==========================================

Demonstrates using ``ax.voxels`` to visualize parts of a color space
'''

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# prepare some coordinates, and attach rgb values to each
x, y, z = np.indices((16, 16, 16))
r = (x + 0.5) / 16
g = (y + 0.5) / 16
b = (z + 0.5) / 16

# define a sphere about [0.5, 0.5, 0.5]
sphere = (r - 0.5)**2 + (g - 0.5)**2 + (b - 0.5)**2 < 0.5**2

# combine the color components
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = r
colors[..., 1] = g
colors[..., 2] = b

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(sphere, color=colors, edgecolor='gray', linewidth=0.5)

plt.show()
