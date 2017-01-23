"""
Demo of downsampling line plots.

This method of downsampling will greatly speed up interaction
and rendering time, without changing the actual rasterization too much.
Try interacting with the two figures (panning, zooming, etc.) to see
the difference.

"""
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(8675309)

# make up some data
y = np.random.normal(loc=0.5, scale=0.4, size=1000000)
x = np.arange(len(y))

# plot without downsampling
plt.figure(1)
plt.plot(x, y)

# plot with downsampling
plt.figure(2)
plt.plot(x, y, downsample=True)

plt.show()

# interact with both figures to compare snapiness
