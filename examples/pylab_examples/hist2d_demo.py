"""
=============
2D Histograms
=============

Demonstrates how to plot 2-dimensional histograms.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

###############################################################################
# Plot a 2D histogram
# -------------------
#
# To plot a 2D histogram, one only needs two vectors of the same length,
# corresponding to each axis of the histogram.

# Fixing random state for reproducibility
np.random.seed(19680801)

# Generate a normal distribution, center at x=0 and y=5
x = np.random.randn(100000)
y = .4 * x + np.random.randn(100000) + 5

fig, ax = plt.subplots()
hist = ax.hist2d(x, y)

###############################################################################
# Customizing your histogram
# --------------------------
#
# Customizing a 2D histogram is similar to the 1D case, you can control
# visual components such as the bin size or color normalization

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# We can increase the number of bins on each axis
axs[0].hist2d(x, y, bins=40)

# As well as define normalization of the colors
axs[1].hist2d(x, y, bins=40, norm=LogNorm())

# We can also define custom numbers of bins for each axis
axs[2].hist2d(x, y, bins=(80, 10), norm=LogNorm())

plt.show()
