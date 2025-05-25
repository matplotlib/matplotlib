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

# Generate random data
x = np.random.random(100).reshape(10, 10)

# Compute histogram
counts, bins = np.histogram(x)

# Set up colormap and normalization
cmap = plt.colormaps['viridis']
norm = mcolors.BoundaryNorm(bins, cmap.N)

fig, ax = plt.subplots()
im = ax.imshow(x, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Value')

# Add histogram as an inset axis
cax = ax.inset_axes([1.05, 0, .25, 1])
midpoints = bins[:-1] + np.diff(bins) / 2
cax.barh(midpoints, counts, height=1/len(counts), color=cmap(norm(midpoints)))
cax.set_yticks(bins)
cax.margins(0)
for spine in cax.spines.values():
    spine.set_visible(False)
cax.set_xlabel('Count')
cax.set_ylabel('Value')

plt.show()

print("Colorbar with histogram example completed.")