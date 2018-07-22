"""
======================
Zoom region inset axes
======================

Example of an inset axes and a rectangle showing where the zoom is located.

"""

import matplotlib.pyplot as plt
import numpy as np


def get_demo_image():
    from matplotlib.cbook import get_sample_data
    import numpy as np
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    Z2 = np.zeros([150, 150], dtype="d")
    ny, nx = z.shape
    Z2[30:30 + ny, 30:30 + nx] = z

    # z is a numpy array of 15x15
    extent = (-3, 4, -4, 3)
    return Z2, extent

fig, ax = plt.subplots(figsize=[5, 4])

# make data
Z2, extent = get_demo_image()
ax.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")
# sub region of the original image
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)
fig.canvas.draw()
plt.show()

#############################################################################
# There is a second interface that closely parallels the interface for
# `~.axes.legend` whereby we specify a location for the inset axes using
# a string code.

fig, ax = plt.subplots(figsize=[5, 4])

ax.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

# inset axes....
axins = ax.inset_axes('NE', width=0.5, height=0.5)

axins.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")
# sub region of the original image
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)
fig.canvas.draw()
plt.show()

#############################################################################
# Its possible to use either form with a transform in data space instead of
# in the axes-relative co-ordinates:

fig, ax = plt.subplots(figsize=[5, 4])

ax.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

# inset axes....
axins = ax.inset_axes([-2.5, 0, 1.6, 1.6], transform=ax.transData)

axins.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")
# sub region of the original image
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)
fig.canvas.draw()
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
matplotlib.axes.Axes.inset_axes
matplotlib.axes.Axes.indicate_inset_zoom
matplotlib.axes.Axes.imshow
