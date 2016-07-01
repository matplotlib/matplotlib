"""
Demo of the streamplot function with starting points.

This example shows how to fix the streamlines that are plotted, by passing
an array of seed points to the `start_points` keyword argument.
"""
import numpy as np
import matplotlib.pyplot as plt

X, Y = (np.linspace(-3, 3, 100),
        np.linspace(-3, 3, 100))

U, V = np.mgrid[-3:3:100j, 0:0:100j]

seed_points = np.array([[-2, 0, 1], [-2, 0, 1]])

fig, ax = plt.subplots()
strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2,
                     cmap=plt.cm.autumn, start_points=seed_points.T)
fig.colorbar(strm.lines)

ax.plot(seed_points[0], seed_points[1], 'bo')

ax.axis((-3, 3, -3, 3))

plt.show()
