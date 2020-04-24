"""
============
3D errorbars
============

An example of using errorbars with upper and lower limits in mplot3d.
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# setting up a parametric curve
t = np.arange(0, 2*np.pi+.1, 0.01)
x, y, z = np.sin(t), np.cos(3*t), np.sin(5*t)

estep = 15
zuplims = [True if (not i % estep and i // estep % 3 == 0)
           else False for i in range(t.size)]
zlolims = [True if (not i % estep and i // estep % 3 == 2)
           else False for i in range(t.size)]

ax.errorbar(x, y, z, 0.2, zuplims=zuplims, zlolims=zlolims, errorevery=estep)

ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")

plt.show()
