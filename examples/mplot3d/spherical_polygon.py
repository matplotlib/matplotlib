from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.spherical import SphericalPolygon
import numpy as np

# plot unit sphere for reference
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='y', alpha=0.1)

# plot spherical triangle
points_triangle = np.array([[0, 1., 0], [0, 0, 1.], [-1., 0, 0]])
sp_triangle = SphericalPolygon(points_triangle)
sp_triangle.add_to_ax(ax, alpha=0.8, color='r')

# plot spherical square
points_square = np.array([[0.7, 1., 1.],
                          [0.5, -1., 1.],
                          [0.5, -1., -1.],
                          [0.5, 1., -1]])
points_square = np.array([p / np.linalg.norm(p) for p in points_square])
tri = np.array([[3, 0, 1], [2, 0, 1], [2, 3, 1], [2, 3, 0]])
# in case tri is not known, you can calculate it using
# scipy.spatial.ConvexHull
sp = SphericalPolygon(points_square, tri)
sp.add_to_ax(ax, alpha=0.8, color='b')

plt.show()
