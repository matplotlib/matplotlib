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

# === Inset Histogram Using ax.inset_axes ===
cax = ax.inset_axes([1.28, 0.05, 0.25, 1.2])

# === Plot Histogram ===
midpoints = bins[:-1] + np.diff(bins) / 2
bar_height = 1 / len(counts)
cax.barh(midpoints, counts, height=bar_height, color=cmap(norm(midpoints)))

# === Remove spines and margins ===
for spine in cax.spines.values():
    spine.set_visible(False)
cax.margins(0)

# Optional: clean up ticks
cax.tick_params(axis='both', which='both', length=0)

# === Labels and ticks ===
cax.set_xlabel('Count', labelpad=8)   # increased from 6 → 8
cax.set_ylabel('Value', labelpad=10)  # increased from 8 → 10
cax.set_yticks(bins)

plt.show()

