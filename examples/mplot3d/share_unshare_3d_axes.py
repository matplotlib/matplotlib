"""
============================================
Parametric Curve with Share and Unshare Axes
============================================

This example demonstrates plotting a parametric curve in 3D,
and how to share and unshare 3D plot axes.
"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z ** 2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

fig = plt.figure()
ax = fig.add_subplot(311, projection='3d')

ax.plot(x, y, z, label='parametric curve')
ax.legend()

ax1 = fig.add_subplot(312)
ax1.plot(range(10))
ax1.share_axes(ax)

ax2 = fig.add_subplot(313, projection='3d', sharex=ax)
ax2.plot(x, y, z)

ax2.unshare_x_axes()
ax2.share_z_axes(ax)

plt.show()
