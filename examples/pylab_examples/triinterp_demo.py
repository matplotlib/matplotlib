"""
Interpolation from triangular grid to quad grid.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


# Create triangulation.
x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5])
y = np.asarray([0, 0, 0, 0, 1,   1,   1,   2, 2, 3])
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
             [5, 6, 8], [5, 8, 7], [7, 8, 9]]
triang = mtri.Triangulation(x, y, triangles)

# Interpolate to regularly-spaced quad grid.
z = np.cos(1.5*x)*np.cos(1.5*y)
interp = mtri.LinearTriInterpolator(triang, z)
xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))
zi = interp(xi, yi)

# Plot the triangulation.
plt.subplot(121)
plt.tricontourf(triang, z)
plt.triplot(triang, 'ko-')
plt.title('Triangular grid')

# Plot interpolation to quad grid.
plt.subplot(122)
plt.contourf(xi, yi, zi)
plt.plot(xi, yi, 'k-', alpha=0.5)
plt.plot(xi.T, yi.T, 'k-', alpha=0.5)
plt.title('Linear interpolation')

plt.show()
