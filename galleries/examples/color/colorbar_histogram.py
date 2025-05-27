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
from matplotlib.ticker import MaxNLocator

# === Surface Data ===
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# === Histogram from actual Z data ===
counts, bins = np.histogram(Z, bins=30)

# === Colormap & Normalization ===
cmap = plt.get_cmap('RdYlBu')
norm = mcolors.BoundaryNorm(bins, cmap.N)

# === Main Plot ===
fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', cmap=cmap,
               origin='lower', extent=[-3, 3, -3, 3],
               norm=norm)

# Adjust image position to allow space
plt.subplots_adjust(right=0.78, top=0.92, bottom=0.08)

# === Inset Histogram – Positioning adjusted ===
cax = ax.inset_axes([1.18, 0.02, 0.25, 0.95])  # left, bottom, width, height

# === Plot Histogram ===
midpoints = bins[:-1] + np.diff(bins) / 2
bar_height = 1 / len(counts)
cax.barh(midpoints, counts, height=np.median(np.diff(bins))*0.8, color=cmap(norm(midpoints)))

# === Clean up ===
#showcase none loop version
cax.spines[:].set_visible(False)
cax.margins(0)
cax.tick_params(axis='both', which='both', length=0)

# === Axis labels ===
cax.set_xlabel('Count', labelpad=10)
cax.set_ylabel('Value', labelpad=6) 

# === Ticks ===
cax.set_yticks(bins)
cax.yaxis.set_major_locator(MaxNLocator(nbins=8))

plt.show()

