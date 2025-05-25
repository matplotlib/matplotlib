"""
=========================
Colorbar with Histogram
=========================

This example demonstrates how to create a colorbar for an image and
add a histogram of the data values alongside it. This is useful for
visualizing the distribution of values mapped to colors.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Generate random data
x = np.random.random((10, 10))

# Compute histogram
counts, bins = np.histogram(x)

# Set up colormap and normalization
cmap = plt.get_cmap('viridis')
norm = mcolors.BoundaryNorm(bins, cmap.N)

fig, ax = plt.subplots()
im = ax.imshow(x, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Value')

# Create an axes on the right side of ax. The width of cax will be 20% of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="20%", pad=0.05)

# Plot histogram
midpoints = bins[:-1] + np.diff(bins) / 2
cax.barh(midpoints, counts, height=np.diff(bins), color=cmap(norm(midpoints)))
cax.set_yticks(bins)
cax.set_xlabel('Count')
cax.set_ylabel('Value')
cax.invert_yaxis()  # Optional: to match the orientation of imshow

plt.tight_layout()
plt.show()

