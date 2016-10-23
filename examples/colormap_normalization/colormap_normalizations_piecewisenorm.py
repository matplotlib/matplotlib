"""
============================================
Examples of arbitrary colormap normalization
============================================

Here I plot an image array with data spanning for a large dynamic range,
using different normalizations. Look at how each of them enhances
different features.

"""


import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from sampledata import PiecewiseNormData

X, Y, data = PiecewiseNormData()
cmap = cm.spectral

# Creating functions for plotting


def make_plot(norm, label=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={
        'width_ratios': [1, 2]}, figsize=plt.figaspect(0.5))
    fig.subplots_adjust(top=0.87, left=0.07, right=0.96)
    fig.suptitle(label)

    cax = ax2.pcolormesh(X, Y, data, cmap=cmap, norm=norm)
    ticks = cax.norm.ticks() if norm else None
    fig.colorbar(cax, format='%.3g', ticks=ticks)
    ax2.set_xlim(X.min(), X.max())
    ax2.set_ylim(Y.min(), Y.max())

    data_values = np.linspace(cax.norm.vmin, cax.norm.vmax, 100)
    cm_values = cax.norm(data_values)
    ax1.plot(data_values, cm_values)
    ax1.set_xlabel('Data values')
    ax1.set_ylabel('Colormap values')


def make_3dplot(label=''):
    fig = plt.figure()
    fig.suptitle(label)
    ax = fig.gca(projection='3d')
    cax = ax.plot_surface(X, Y, data, rstride=1, cstride=1,
                          cmap=cmap, linewidth=0, antialiased=False)
    ax.set_zlim(data.min(), data.max())
    fig.colorbar(cax, shrink=0.5, aspect=5)
    ax.view_init(20, 225)


# Showing how the data looks in linear scale
make_3dplot('Regular linear scale')
make_plot(None, 'Regular linear scale')

# Example of logarithm normalization using FuncNorm
norm = colors.FuncNorm(f=lambda x: np.log10(x),
                       finv=lambda x: 10.**(x), vmin=0.01, vmax=2)
make_plot(norm, "Log normalization using FuncNorm")
# The same can be achived with
# norm = colors.FuncNorm(f='log',vmin=0.01,vmax=2)

# Example of root normalization using FuncNorm
norm = colors.FuncNorm(f='sqrt', vmin=0.0, vmax=2)
make_plot(norm, "Root normalization using FuncNorm")

# Performing a symmetric amplification of the features around 0
norm = colors.MirrorPiecewiseNorm(fpos='crt')
make_plot(norm, "Amplified features symetrically around \n"
                "0 with MirrorPiecewiseNorm")


# Amplifying features near 0.6 with MirrorPiecewiseNorm
norm = colors.MirrorPiecewiseNorm(fpos='crt', fneg='crt',
                                  center_cm=0.35,
                                  center_data=0.6)
make_plot(norm, "Amplifying positive and negative features\n"
                "standing on 0.6 with MirrorPiecewiseNorm")

# Amplifying features near both -0.4 and near 1.2 with PiecewiseNorm
norm = colors.PiecewiseNorm(flist=['cubic', 'crt', 'cubic', 'crt'],
                            refpoints_cm=[0.25, 0.5, 0.75],
                            refpoints_data=[-0.4, 1, 1.2])
make_plot(norm, "Amplifying positive and negative features standing\n"
                " on -0.4 and 1.2 with PiecewiseNorm")

# Amplifying features near both -1, -0.2 and near 1.2 with PiecewiseNorm
norm = colors.PiecewiseNorm(flist=['crt', 'crt', 'crt'],
                            refpoints_cm=[0.4, 0.7],
                            refpoints_data=[-0.2, 1.2])
make_plot(norm, "Amplifying only positive features standing on -1, -0.2\n"
                " and 1.2 with PiecewiseNorm")


plt.show()
