"""
==========
Streamplot
==========

A stream plot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the `~.axes.Axes.streamplot` function:

* Varying the color along a streamline.
* Varying the density of streamlines.
* Varying the line width along a streamline.
* Controlling the starting points of streamlines.
* Streamlines skipping masked regions and NaN values.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U**2 + V**2)


gs = {'height_ratios': [1, 1, 2]}
fig, axs = plt.subplots(nrows=3, ncols=2, gridspec_kw=gs, figsize=(7, 9))

#  Varying density along a streamline
axs[0, 0].streamplot(X, Y, U, V, density=[0.5, 1])
axs[0, 0].set_title('Varying Density')

# Varying color along a streamline
strm = axs[0, 1].streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
fig.colorbar(strm.lines, ax=axs[0, 1])
axs[0, 1].set_title('Varying Color')

#  Varying line width along a streamline
lw = 5*speed / speed.max()
axs[1, 0].streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)
axs[1, 0].set_title('Varying Line Width')

# Controlling the starting points of the streamlines
seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

strm = axs[1, 1].streamplot(X, Y, U, V, color=U, linewidth=2,
                            cmap='autumn', start_points=seed_points.T)
fig.colorbar(strm.lines, ax=axs[1, 1])
axs[1, 1].set_title('Controlling Starting Points')

# Displaying the starting points with blue symbols.
axs[1, 1].plot(seed_points[0], seed_points[1], 'bo')
axs[1, 1].set(xlim=(-w, w), ylim=(-w, w))

# Create a mask
mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = True
U[:20, :20] = np.nan
U = np.ma.array(U, mask=mask)

gs = axs[0, 0].get_subplotspec().get_gridspec()
axs[2, 0].remove()
axs[2, 1].remove()
ax4 = fig.add_subplot(gs[2:, :])
ax4.streamplot(X, Y, U, V, color='r')
ax4.set_title('Streamplot with Masking')

ax4.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5, cmap='gray', aspect='auto')
ax4.set_aspect('equal')

plt.tight_layout()
plt.show()
#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions and methods is shown in this example:

import matplotlib
matplotlib.axes.Axes.streamplot
matplotlib.pyplot.streamplot
matplotlib.gridspec
matplotlib.gridspec.GridSpec
