"""
========================================
3D Histogram with hexagonal bins
========================================

Demonstrates visualising a 3D density map of data using hexagonal tessellation.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cbook import hexbin

# Fixing random state for reproducibility
np.random.seed(42)

# Generate samples from mltivariate Gaussian
# Parameters
mu = (0, 0)
sigma = ([0.8, 0.3],
         [0.3, 0.5])
n = 10_000
gridsize = 15
# draw samples
xy = np.random.multivariate_normal(mu, sigma, n)
# histogram samples with hexbin
xyz, (xmin, xmax), (ymin, ymax), (nx, ny) = hexbin(*xy.T, gridsize=gridsize,
                                                   mincnt=3)
# compute bar cross section size
dxy = np.array([(xmax - xmin) / nx, (ymax - ymin) / ny / np.sqrt(3)]) * 0.95

# plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.hexbar3d(*xyz, dxy, cmap='plasma')
ax.set(xlabel='x', ylabel='y', zlabel='z')

plt.show()
