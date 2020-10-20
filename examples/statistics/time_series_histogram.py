"""
=====================
Time Series Histogram
=====================

This example demonstrates how to efficiently visualize large numbers of time
series in a way that could potentially reveal hidden substructure and patterns
that are not immediately obvious.

The first plot shows the typical way of visualizing multiple time series by
overlaying them on top of each other with ``plt.plot``. The second and third
plots show how to reinterpret the data as a 2d histogram.
"""
from copy import copy
import time

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_, axes = plt.subplots(nrows=3, figsize=(10, 6 * 3))

# Make some data; a 1D random walk + small fraction of sine waves
num_series = 10000
num_points = 100
SNR = 0.10  # Signal to Noise Ratio
x = np.linspace(0, 4 * np.pi, num_points)
# random walk
Y = np.cumsum(np.random.randn(num_series, num_points), axis=-1)
# sinusoidal signal
num_signal = int(round(SNR * num_series))
phi = (np.pi / 8) * np.random.randn(num_signal, 1)  # small random offest
Y[-num_signal:] = ((
    np.sqrt(np.arange(num_points))[None, :]
    * np.sin(x[None, :] - phi))
    + 0.05 * np.random.randn(num_signal, num_points)
)

# Plot it using `plot` and the lowest nonzero value of alpha (1/256).
# With this view it is extremely difficult to observe the sinusoidal behavior
# because of how many overlapping series there are. It also takes some time
# to run because so many individual plots that need to be generated.
tic = time.time()
for i in range(Y.shape[0]):
    axes[0].plot(x, Y[i], color="C0", alpha=1 / 256)
toc = time.time()
axes[0].set_title(
    r"Standard time series visualization using line plot")
print(f"{toc-tic:.2f} sec. elapsed")  # ~4 seconds


# Now we will convert the multiple time series into a heat map. Not only will
# the hidden signal be more visible, but it is also a much quicker procedure.
tic = time.time()
# linearly interpolate between the points in each time series
num_fine = 400 * 3
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.zeros((num_series, num_fine))
for i in range(num_series):
    y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
y_fine = y_fine.flatten()
x_fine = np.matlib.repmat(x_fine, num_series, 1).flatten()


# Plot (x, y) points in 2d histogram with log colorscale
# It is pretty evident that there is some kind of structure under the noise
cmap = copy(plt.cm.jet)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm())
axes[1].set_title(
    r"Alternative time series vis. using 2d histogram and log color scale")

# Same thing on linear color scale but with different (more visible) cmap
cmap = copy(plt.cm.Blues)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
# tune vmax to make signal more visible
axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap, vmax=3e3)
axes[2].set_title(
    r"Alternative time series vis. using 2d histogram and linear color scale")
toc = time.time()
print(f"{toc-tic:.2f} sec. elapsed")  # ~1 sec for both plots + interpolation

plt.show()
