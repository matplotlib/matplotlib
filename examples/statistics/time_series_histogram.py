"""
=====================
Time Series Histogram
=====================

This example demonstrates how to efficiently visualize large numbers of time
series in a way that could potentially reveal hidden substructure and patterns
that are not immediately obvious.

The first plot shows the typical way of visualizing multiple time series by
overlaying them on top of each other with `plt.plot`. The second and third
plots show how to reinterpret the data as a 2d histogram.
"""
from copy import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_, axes = plt.subplots(nrows=3, figsize=(10, 6 * 3))

# Make some data; lots of random noise + small fraction of sine waves
num_series = 10000
num_points = 100
SNR = 0.05  # Signal to Noise Ratio
x = np.linspace(0, 4 * np.pi, num_points)
# random noise
Y = np.random.randn(num_series, num_points)
# sinusoidal signal
num_signal = int(round(SNR * num_series))
phi = (0.25 * np.pi) * np.random.randn(num_signal, 1)
Y[-num_signal:] = np.sin(x[None, :] - phi) + 0.1 * \
    np.random.randn(num_signal, num_points)

# Plot it using `plot` and the lowest nonzero value of alpha (1/256).
# With this view it is extremely difficult to observe the sinusoidal behavior
# because of how many overlapping series there are. It also takes some time
# to run because so many individual plots that need to be generated.
tic = time.time()
for i in range(Y.shape[0]):
    axes[0].plot(x, Y[i], color="C0", alpha=1 / 256)
toc = time.time()
axes[0].set_title(
    r"Standard time series visualization using `plt.plot`")
print(f"{toc-tic:.2f} sec. elapsed")  # ~4 seconds


# Now we will convert the multiple time series into a heat map. Not only will
# the hidden signal be more visible, but it is also a much quicker procedure.
tic = time.time()
# linearly interpolate between the points in each time series
num_fine = 1000
x_fine = np.linspace(x.min(), x.max(), num_fine)  # x_fine.shape == (1_000,)
y_fine = np.stack([np.interp(x_fine, x, Y[i]) for i in range(
    Y.shape[0])], axis=0)  # y_fine.shape = (10_000, 1_000)
# convert into tensor of (x, y) pairs along the -1 axis
xy = np.stack([np.broadcast_to(x_fine[None, :], y_fine.shape),
               y_fine], axis=-1)  # xy.shape == (10_000, 1_000, 2)
xy = xy.reshape(-1, 2)  # xy.shape = (10_000_000, 2)

# Plot (x, y) points in 2d histogram with log colorscale
# It is pretty evident that there is some kind of structure under the noise
# that has a periodicity of about ~6 and oscillates between +1/-1.
cmap = copy(plt.cm.Blues)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(*xy.T, bins=[200, 200])
axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm())
axes[1].set_title(
    r"Alternative time series vis. using `plt.hist2d` and log color scale")

# It is even visible on a linear color scale
h, xedges, yedges = np.histogram2d(*xy.T, bins=[200, 200])
axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap)
axes[2].set_title(
    r"Alternative time series vis. using `plt.hist2d` and linear color scale")
toc = time.time()
print(f"{toc-tic:.2f} sec. elapsed")  # ~1 sec for both plots

plt.show()
