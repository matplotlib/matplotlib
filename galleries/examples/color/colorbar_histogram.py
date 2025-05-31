"""
=====================
Histogram as colorbar
=====================

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
x = y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-((X+1)*1.3)**2 - ((Y+1)*1.3)**2)
Z2 = 2.5*np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1**0.25 - Z2**0.5)

# === Histogram from actual Z data ===
counts, bins = np.histogram(Z, bins=30)

# === Colormap & Normalization ===
cmap = plt.get_cmap('RdYlBu_r')
norm = mcolors.BoundaryNorm(bins, cmap.N)

# === Main Plot ===
fig, ax = plt.subplots(layout="constrained")
im = ax.imshow(Z, cmap=cmap,
               origin='lower', extent=[-3, 3, -3, 3],
               norm=norm)

# === Inset Histogram – Positioning adjusted ===
cax = ax.inset_axes([1.18, 0.02, 0.25, 0.95])  # left, bottom, width, height

# === Plot Histogram ===
midpoints = bins[:-1] + np.diff(bins) / 2
height = np.median(np.diff(bins)) * 0.8
colors = cmap(norm(midpoints))

cax.barh(midpoints, counts, height=height, color=colors)

# === Clean up ===
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

