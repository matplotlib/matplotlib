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
    * Streamlines skipping masked regions and NaN values.
    * Controlling the start points of streamlines.
"""
import numpy as np
import matplotlib.pyplot as plt

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

# Varying color along a streamline
fig0, ax = plt.subplots()
strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
fig0.colorbar(strm.lines)

# Varying the density of streamlines
fig1, (ax1, ax2) = plt.subplots(ncols=2)
ax1.streamplot(X, Y, U, V, density=[0.5, 1])

# Varying the line width along a stream line
lw = 5*speed / speed.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

# Streamlines skipping masked regions and NaN values
mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = True
U[:20, :20] = np.nan
U = np.ma.array(U, mask=mask)

fig2, ax = plt.subplots()
ax.streamplot(X, Y, U, V, color='r')

ax.imshow(-mask, extent=(-w, w, -w, w), alpha=0.5,
          interpolation='nearest', cmap=plt.cm.gray)

# Controlling the start points of streamlines
X, Y = (np.linspace(-3, 3, 100),
        np.linspace(-3, 3, 100))

U, V = np.mgrid[-3:3:100j, 0:0:100j]

seed_points = np.array([[-2, 0, 1], [-2, 0, 1]])

fig3, ax = plt.subplots()
strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2,
                     cmap="autumn", start_points=seed_points.T)
fig3.colorbar(strm.lines)

ax.plot(seed_points[0], seed_points[1],'bo')

ax.axis((-3, 3, -3, 3))

plt.show()
