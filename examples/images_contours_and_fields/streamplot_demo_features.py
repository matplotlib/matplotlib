"""
==========
Streamplot
==========

Demo of the `streamplot` function.

A streamplot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the stream plot function:

    * Varying the color along a streamline.
    * Varying the density of streamlines.
    * Varying the line width along a stream line.
"""
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)

fig1, (ax1, ax2) = plt.subplots(ncols=2)
ax1.streamplot(X, Y, U, V, density=[0.5, 1])

lw = 5*speed / speed.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()

"""
================================
Streamplot function with masking
================================

This example shows how streamlines created by the streamplot function skips
masked regions and NaN values.
"""
w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = True
U[:20, :20] = np.nan
U = np.ma.array(U, mask=mask)

fig, ax = plt.subplots()
ax.streamplot(X, Y, U, V, color='r')

ax.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5,
          interpolation='nearest', cmap=plt.cm.gray)

plt.show()

"""
========================================
Streamplot function with starting points
========================================

This example shows how to fix the streamlines that are plotted, by passing
an array of seed points to the `start_points` keyword argument.
"""
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2

# 5 points along the first diagonal and a point in the left upper quadrant
seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

fig, ax = plt.subplots()
strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2,
                     cmap=plt.cm.autumn, start_points=seed_points.T)
fig.colorbar(strm.lines)

ax.plot(seed_points[0], seed_points[1], 'bo')

ax.axis((-3, 3, -3, 3))

plt.show()
