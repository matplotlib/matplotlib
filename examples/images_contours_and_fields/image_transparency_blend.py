"""
=============================================
Blend transparency with color with 2-D images
=============================================

Blend transparency with color to highlight parts of data with imshow.

A common use for :func:`matplotlib.pyplot.imshow` is to plot a 2-D statistical
map. In this case, it's common to visualize the statistic of choice (e.g.,
a t-statistic) alongisde another value of interest (e.g., the p-value for that
statistic). One way to do this is to map the p-value onto the transparency of
the image such that data with "significant" values are highlighted.

This example demonstrates how you can achieve this effect using
:class:`matplotlib.colors.Normalize. Note that it is not possible to directly
pass alpha values to :func:`matplotlib.pyplot.imshow`.

First we will generate some data, in this case, we'll create two 2-D "blobs"
in a 2-D grid. One blob will be positive, and the other negative.
"""
# sphinx_gallery_thumbnail_number = 2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Generate the space in which the blobs will live
xmin, xmax, ymin, ymax = (0, 100, 0, 100)
xx = np.linspace(xmin, xmax, 100)
yy = np.linspace(ymin, ymax, 100)
grid = np.array(np.meshgrid(xx, yy))
grid = grid.transpose(2, 1, 0)

# Generate the blobs
means_high = [20, 50]
means_low = [50, 60]
cov = [[500, 0], [0, 800]]
gauss_high = multivariate_normal.pdf(grid, means_high, cov)
gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
weights = gauss_high + gauss_low

# We'll plot these blobs using only ``imshow``.
vmax = np.abs(weights).max() * 1.5
vmin = -vmax
cmap = plt.cm.RdYlBu
fig, ax = plt.subplots()
ax.imshow(weights, extent=(xmin, xmax, ymin, ymax), cmap=cmap)
ax.set_axis_off()

################################################################################
# Blending in transparency
# ========================
#
# Below, we'll recreate the same plot, but this time we'll blend in
# transparency with the image so that the extreme values are highlighted.
# We'll also add in contour lines to highlight the image values.

# Create an alpha channel based on weight values
alphas = Normalize(0, .0001, clip=True)(np.abs(weights))
alphas = np.clip(alphas, .4, 1)

# Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
colors = Normalize(vmin, vmax)(weights)
colors = cmap(colors)

# Now set the alpha channel to the one we created above
colors[..., -1] = alphas

# Create the figure and image
# Note that the absolute values may be slightly different
fig, ax = plt.subplots()
ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))

# Add contour lines to further highlight different levels.
ax.contour(weights[::-1], levels=[-.0001, .0001], colors='k', linestyles='-')
ax.set_axis_off()
plt.show()
