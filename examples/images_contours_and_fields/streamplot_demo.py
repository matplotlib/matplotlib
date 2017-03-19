"""
==========
Streamplot
==========
A streamplot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the stream plot function:

    * Varying the color along a streamline.
    * Varying the density of streamlines.
    * Varying the line width along a stream line.
    * Controlling the start points of streamlines.
    * Streamlines skipping masked regions and NaN values.
"""
import numpy as np
import matplotlib.pyplot as plt

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

fig = plt.figure()

#  Varying density along a streamline
ax0 = fig.add_subplot(321)
ax0.streamplot(X, Y, U, V, density=[0.5, 1])
ax0.set_title('Varying Density')

# Varying color along a streamline
ax1 = fig.add_subplot(322)
strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
fig.colorbar(strm.lines)
ax1.set_title('Varying color')

#  Varying line width along a streamline
ax2 = fig.add_subplot(323)
lw = 5*speed / speed.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)
ax2.set_title('Varying Line Width')

# Controlling the starting points of the streamlines
X, Y = (np.linspace(-3, 3, 100),
        np.linspace(-3, 3, 100))
U, V = np.mgrid[-3:3:100j, 0:0:100j]
seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

ax3 = fig.add_subplot(324)
strm = ax3.streamplot(X, Y, U, V, color=U, linewidth=2,
                     cmap=plt.cm.autumn, start_points=seed_points.T)
fig.colorbar(strm.lines)
ax3.set_title('Controlling Starting Points')

# Displaying the starting points with red symbols.
ax3.plot(seed_points[0], seed_points[1], 'bo')

ax3.axis((-3, 3, -3, 3))

# Create a mask
w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = True
U[:20, :20] = np.nan
U = np.ma.array(U, mask=mask)

ax4 = fig.add_subplot(325)
ax4.streamplot(X, Y, U, V, color='r')
ax4.set_title('Streamline with Masking')

ax4.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5,
          interpolation='nearest', cmap=plt.cm.gray, aspect='auto')

plt.tight_layout()
plt.show()
