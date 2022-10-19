"""
===========================
Colored triangular 3D surfaces
===========================

This shows how to do colored trisurf plots.

"""

import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

fig = plt.figure()

# Create a parametric sphere
r = np.linspace(0, np.pi, 50)
phi = np.linspace(-np.pi, np.pi, 50)
r, phi = np.meshgrid(r, phi)
r, phi = r.flatten(), phi.flatten()
tri = mtri.Triangulation(r, phi)

x = np.sin(r)*np.cos(phi)
y = np.sin(r)*np.sin(phi)
z = np.cos(r)

ax = fig.add_subplot(projection='3d')
ax.plot_trisurf(x, y, z,
                triangles=tri.triangles,
                cmap=plt.colormaps['terrain'],
                C=np.sin(2*phi)*(r - np.pi/2),
                shade=True)

fig.tight_layout()
plt.show()
