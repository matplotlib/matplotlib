"""
=====================================================================
Examples of normalization using  :class:`~matplotlib.colors.FuncNorm`
=====================================================================

This is an example on how to perform a normalization using an arbitrary
function with :class:`~matplotlib.colors.FuncNorm`. A logarithm normalization
and a square root normalization will be use as examples.

"""

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import numpy as np

norm_log = colors.FuncNorm(f='log10', vmin=0.01)
# The same can be achieved with
# norm_log = colors.FuncNorm(f=np.log10,
#                            finv=lambda x: 10.**(x), vmin=0.01)

norm_sqrt = colors.FuncNorm(f='sqrt', vmin=0.0)
# The same can be achieved with
# norm_sqrt = colors.FuncNorm(f='root{2}', vmin=0.)
# or with
# norm_sqrt = colors.FuncNorm(f=lambda x: x**0.5,
#                             finv=lambda x: x**2, vmin=0.0)

normalizations = [(None, 'Regular linear scale'),
                  (norm_log, 'Log normalization'),
                  (norm_sqrt, 'Root normalization')]

# Fabricating some data
x = np.linspace(0, 1, 300)
y = np.linspace(-1, 1, 90)
X, Y = np.meshgrid(x, y)

data = np.zeros(X.shape)


def gauss2d(x, y, a0, x0, y0, wx, wy):
    return a0 * np.exp(-(x - x0)**2 / wx**2 - (y - y0)**2 / wy**2)

for x in np.linspace(0., 1, 15):
    data += gauss2d(X, Y, x, x, 0, 0.25 / 15, 0.25)

data -= data.min()
data /= data.max()

# Using the custom normalizations to plot the data
fig, axes = plt.subplots(3, 2, sharex='col',
                         gridspec_kw={'width_ratios': [1, 3.5]},
                         figsize=plt.figaspect(0.6))

for (ax_left, ax_right), (norm, title) in zip(axes, normalizations):

    # Showing the normalization effect on an image
    cax = ax_right.imshow(data, cmap=cm.afmhot, norm=norm, aspect='auto')
    fig.colorbar(cax, format='%.3g', ax=ax_right)
    ax_right.set_title(title)
    ax_right.xaxis.set_ticks([])
    ax_right.yaxis.set_ticks([])

    # Plotting the behaviour of the normalization
    d_values = np.linspace(cax.norm.vmin, cax.norm.vmax, 100)
    cm_values = cax.norm(d_values)
    ax_left.plot(d_values, cm_values)
    ax_left.set_ylabel('Colormap values')

ax_left.set_xlabel('Data values')

plt.show()
