import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

def plot_data(ax):
    im = ax.imshow(Z, interpolation='bilinear', cmap='jet',
                   origin='lower', extent=[-3,3,-3,3],
                   vmax=abs(Z).max(), vmin=-abs(Z).max())
    return im

# Display the same plot 4 different ways:
#  - no filter
#  - luminosity filter
#  - Deuteranope color blindness filter
#  - Tritanope color blindness filter

ax = plt.subplot(2, 2, 1)
plot_data(ax)
ax.set_title("Original")

ax = plt.subplot(2, 2, 2)
im = plot_data(ax)
im.set_agg_filter('luminosity')
ax.set_title("Luminosity")

ax = plt.subplot(2, 2, 3)
im = plot_data(ax)
im.set_agg_filter("deuteranope")
ax.set_title("Deuteranope")

ax = plt.subplot(2, 2, 4)
im = plot_data(ax)
im.set_agg_filter("tritanope")
ax.set_title("Tritanope")

plt.show()
