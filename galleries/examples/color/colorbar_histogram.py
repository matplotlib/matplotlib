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

# === Surface Data ===
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# Purposeful bins for full value range
bins = np.linspace(-1, 1, 11)
counts, _ = np.histogram(Z, bins=bins)

# Normalize to bins for colormap
cmap = plt.get_cmap('RdYlBu')
norm = mcolors.BoundaryNorm(bins, cmap.N)

# === Plot heatmap ===
fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', cmap=cmap,
               origin='lower', extent=[-3, 3, -3, 3],
               norm=norm)

# === Inset histogram as colorbar ===
cax = inset_axes(ax,
                 width="20%",
                 height="95%",
                 loc='center left',
                 bbox_to_anchor=(1.18, 0.025, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0)

# Histogram values and bars
midpoints = bins[:-1] + np.diff(bins) / 2
bar_height = np.min(np.diff(bins))
cax.barh(midpoints, counts, height=bar_height, color=cmap(norm(midpoints)))
cax.margins(0)

# Axis labels with clean spacing
cax.set_xlabel('Count', labelpad=6)
cax.set_ylabel('Value', labelpad=2)

# Use bin edges as y-ticks (no inversion now)
cax.set_yticks(bins)

# Leave room for histogram inset
plt.subplots_adjust(right=0.75)

plt.show()