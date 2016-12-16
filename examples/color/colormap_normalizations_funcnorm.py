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


def main():
    fig, axes = plt.subplots(3, 2, gridspec_kw={
              'width_ratios': [1, 3.5]}, figsize=plt.figaspect(0.6))

    # Example of logarithm normalization using FuncNorm
    norm_log = colors.FuncNorm(f='log10', vmin=0.01)
    # The same can be achieved with
    # norm_log = colors.FuncNorm(f=np.log10,
    #                            finv=lambda x: 10.**(x), vmin=0.01)

    # Example of root normalization using FuncNorm
    norm_sqrt = colors.FuncNorm(f='sqrt', vmin=0.0)
    # The same can be achieved with
    # norm_sqrt = colors.FuncNorm(f='root{2}', vmin=0.)
    # or with
    # norm_sqrt = colors.FuncNorm(f=lambda x: x**0.5,
    #                             finv=lambda x: x**2, vmin=0.0)

    normalizations = [(None, 'Regular linear scale'),
                      (norm_log, 'Log normalization'),
                      (norm_sqrt, 'Root normalization')]

    for i, (norm, title) in enumerate(normalizations):
        X, Y, data = get_data()

        # Showing the normalization effect on an image
        ax2 = axes[i][1]
        cax = ax2.imshow(data, cmap=cm.afmhot, norm=norm)
        ticks = cax.norm.ticks(5) if norm else np.linspace(0, 1, 6)
        fig.colorbar(cax, format='%.3g', ticks=ticks, ax=ax2)
        ax2.set_title(title)
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])

        # Plotting the behaviour of the normalization
        ax1 = axes[i][0]
        d_values = np.linspace(cax.norm.vmin, cax.norm.vmax, 100)
        cm_values = cax.norm(d_values)
        ax1.plot(d_values, cm_values)
        ax1.set_xlabel('Data values')
        ax1.set_ylabel('Colormap values')

    plt.show()


def get_data(_cache=[]):
    if len(_cache) > 0:
        return _cache[0]
    x = np.linspace(0, 1, 300)
    y = np.linspace(-1, 1, 90)
    X, Y = np.meshgrid(x, y)

    data = np.zeros(X.shape)

    def gauss2d(x, y, a0, x0, y0, wx, wy):
        return a0 * np.exp(-(x - x0)**2 / wx**2 - (y - y0)**2 / wy**2)
    N = 15
    for x in np.linspace(0., 1, N):
        data += gauss2d(X, Y, x, x, 0, 0.25 / N, 0.25)

    data = data - data.min()
    data = data / data.max()
    _cache.append((X, Y, data))

    return _cache[0]

main()
