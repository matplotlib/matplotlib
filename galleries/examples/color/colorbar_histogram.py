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
from matplotlib import gridspec

# Generate random data
x = np.random.random(100).reshape(10, 10)

# Compute histogram
counts, bins = np.histogram(x)

# Set up colormap and normalization
cmap = plt.colormaps['viridis']
norm = mcolors.BoundaryNorm(bins, cmap.N)

# Create figure with constrained_layout for better spacing
fig = plt.figure(figsize=(8, 4), constrained_layout=True)
gs = gridspec.GridSpec(1, 3, width_ratios=[4, 0.2, 1], figure=fig)

# Main image
ax_img = fig.add_subplot(gs[0])
im = ax_img.imshow(x, cmap=cmap, norm=norm)
ax_img.set_title("Image")

# Colorbar
cax = fig.add_subplot(gs[1])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Value')

# Histogram
ax_hist = fig.add_subplot(gs[2])
midpoints = bins[:-1] + np.diff(bins) / 2
ax_hist.barh(midpoints, counts, height=np.diff(bins), color=cmap(norm(midpoints)), edgecolor='k')
ax_hist.set_yticks(bins)
ax_hist.set_xlabel('Count')
ax_hist.set_ylabel('Value')
ax_hist.margins(0)
for spine in ax_hist.spines.values():
    spine.set_visible(False)

plt.show()

print("Colorbar with histogram example completed.")
