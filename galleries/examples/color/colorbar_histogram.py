"""
=========================
Histogram as Colorbar
=========================

This example demonstrates how to create a colorbar for an image and
add a histogram of the data values alongside it. This is useful for
visualizing the distribution of values mapped to colors.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Generate random data
x = np.random.random((10, 10))

# Compute histogram
counts, bins = np.histogram(x)

# Set up colormap and normalization
cmap = plt.get_cmap('viridis')
norm = mcolors.BoundaryNorm(bins, cmap.N)

fig, ax = plt.subplots()
im = ax.imshow(x, cmap=cmap, norm=norm)

# Create an inset axes for the histogram
cax = inset_axes(ax,
                 width="20%",    
                 height="95%",   
                 loc='center left',
                 bbox_to_anchor=(1.18, 0.025, 1, 1),  # x-shift, y-shift, width, height
                 bbox_transform=ax.transAxes,
                 borderpad=0)

# Plot histogram
midpoints = bins[:-1] + np.diff(bins) / 2
bar_height = np.min(np.diff(bins))
cax.barh(midpoints, counts, height=bar_height, color=cmap(norm(midpoints)))

# Adjust label distances more precisely
cax.set_xlabel('Count', labelpad=3)    
cax.set_ylabel('Value', labelpad=2)  


cax.set_yticks(bins)
cax.invert_yaxis()  # Optional: match image orientation

# Leave room for histogram inset
plt.subplots_adjust(right=0.75)
plt.show()
