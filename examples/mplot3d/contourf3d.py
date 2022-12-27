"""
===============
Filled contours
===============

`.Axes3D.contourf` differs from `.Axes3D.contour` in that it creates filled
contours, i.e. a discrete number of colours are used to shade the domain.

This is like a `.Axes.contourf` plot in 2D except that the shaded region
corresponding to the level c is graphed on the plane ``z=c``.
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.contourf(X, Y, Z, cmap=cm.coolwarm)

plt.show()
