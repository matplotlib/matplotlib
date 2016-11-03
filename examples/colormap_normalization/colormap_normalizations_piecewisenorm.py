"""
=========================================================================
Examples of normalization using :class:`~matplotlib.colors.PiecewiseNorm`
=========================================================================

This is an example on how to perform a normalization defined by intervals
using class:`~matplotlib.colors.PiecewiseNorm`.

"""

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import numpy as np


def main():
    fig, ((ax11, ax12),
          (ax21, ax22)) = plt.subplots(2, 2, gridspec_kw={
              'width_ratios': [1, 3]}, figsize=plt.figaspect(0.6))

    cax = make_plot(None, 'Regular linear scale', fig, ax11, ax12)
    fig.colorbar(cax, format='%.3g', ax=ax12, ticks=np.linspace(0, 1, 6))

    # Example of amplification of features above 0.2 and 0.6
    norm = colors.PiecewiseNorm(flist=['linear', 'root{4}', 'linear',
                                       'root{4}', 'linear'],
                                refpoints_cm=[0.2, 0.4, 0.6, 0.8],
                                refpoints_data=[0.2, 0.4, 0.6, 0.8])
    cax = make_plot(norm, 'Amplification of features above 0.2 and 0.6',
                    fig, ax21, ax22)
    fig.colorbar(cax, format='%.3g', ticks=cax.norm.ticks(11), ax=ax22)
    # The same can be achieved with
    # norm = colors.PiecewiseNorm(flist=[lambda x: x,
    #                                    lambda x: x**(1. / 4),
    #                                    lambda x: x,
    #                                    lambda x: x**(1. / 4),
    #                                    lambda x: x],
    #                             finvlist=[lambda x: x,
    #                                       lambda x: x**4,
    #                                       lambda x: x,
    #                                       lambda x: x**4,
    #                                       lambda x: x],
    #                             refpoints_cm=[0.2, 0.4, 0.6, 0.8],
    #                             refpoints_data=[0.2, 0.4, 0.6, 0.8])

    fig.subplots_adjust(hspace=0.4, wspace=0.15)
    fig.suptitle('Normalization with PiecewiseNorm')
    plt.show()


def make_plot(norm, label, fig, ax1, ax2):
    X, Y, data = get_data()
    cax = ax2.imshow(data, cmap=cm.gist_heat, norm=norm)

    d_values = np.linspace(cax.norm.vmin, cax.norm.vmax, 300)
    cm_values = cax.norm(d_values)
    ax1.plot(d_values, cm_values)
    ax1.set_xlabel('Data values')
    ax1.set_ylabel('Colormap values')
    ax2.set_title(label)
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    return cax


def get_data(_cache=[]):
    if len(_cache) > 0:
        return _cache[0]
    x = np.linspace(0, 1, 301)[:-1]
    y = np.linspace(-1, 1, 120)
    X, Y = np.meshgrid(x, y)

    data = np.zeros(X.shape)

    def supergauss2d(o, x, y, a0, x0, y0, wx, wy):
        x_ax = ((x - x0) / wx)**2
        y_ax = ((y - y0) / wy)**2
        return a0 * np.exp(-(x_ax + y_ax)**o)
    N = 6

    data += np.floor(X * (N)) / (N - 1)

    for x in np.linspace(0., 1, N + 1)[0:-1]:
        data += supergauss2d(3, X, Y, 0.05, x + 0.5 / N, -0.5, 0.25 / N, 0.15)
        data -= supergauss2d(3, X, Y, 0.05, x + 0.5 / N, 0.5, 0.25 / N, 0.15)

    data = np.clip(data, 0, 1)
    _cache.append((X, Y, data))
    return _cache[0]

main()
