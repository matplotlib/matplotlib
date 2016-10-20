"""
============================================
Examples of arbitrary colormap normalization
============================================

Here I plot an image array with data spanning for a large dynamic range,
using different normalizations. Look at how each of them enhances
different features.

"""

import ArbitraryNorm as colors

import numpy as np
# import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Creating some toy data
xmax = 16 * np.pi
x = np.linspace(0, xmax, 1024)
y = np.linspace(-2, 2, 512)
X, Y = np.meshgrid(x, y)

data = np.zeros(X.shape)


def gauss2d(x, y, a0, x0, y0, wx, wy):
    return a0 * np.exp(-(x - x0)**2 / wx**2 - (y - y0)**2 / wy**2)

maskY = (Y > -1) * (Y <= 0)
N = 31
for i in range(N):
    maskX = (X > (i * (xmax / N))) * (X <= ((i + 1) * (xmax / N)))
    mask = maskX * maskY
    data[mask] += gauss2d(X[mask], Y[mask], 2. * i / (N - 1), (i + 0.5) *
                          (xmax / N), -0.25, xmax / (3 * N), 0.07)
    data[mask] -= gauss2d(X[mask], Y[mask], 1. * i / (N - 1), (i + 0.5) *
                          (xmax / N), -0.75, xmax / (3 * N), 0.07)

maskY = (Y > 0) * (Y <= 1)
data[maskY] = np.cos(X[maskY]) * Y[maskY]**2

N = 61
maskY = (Y > 1) * (Y <= 2.)
for i, val in enumerate(np.linspace(-1, 1, N)):
    if val < 0:
        aux = val
    if val > 0:
        aux = val * 2

    maskX = (X > (i * (xmax / N))) * (X <= ((i + 1) * (xmax / N)))
    data[maskX * maskY] = aux

N = 11
maskY = (Y <= -1)
for i, val in enumerate(np.linspace(-1, 1, N)):
    if val < 0:
        factor = 1
    if val >= 0:
        factor = 2
    maskX = (X > (i * (xmax / N))) * (X <= ((i + 1) * (xmax / N)))
    mask = maskX * maskY
    data[mask] = val * factor

    if i != N - 1:
        data[mask] += gauss2d(X[mask], Y[mask], 0.03 * factor, (i + 0.5) *
                              (xmax / N), -1.25, xmax / (3 * N), 0.07)
    if i != 0:
        data[mask] -= gauss2d(X[mask], Y[mask], 0.1 * factor, (i + 0.5) *
                              (xmax / N), -1.75, xmax / (3 * N), 0.07)


cmap = cm.spectral


def makePlot(norm, label=''):
    fig, ax = plt.subplots()
    cax = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm)
    ax.set_title(label)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-2, 2)
    if norm:
        ticks = norm.ticks()
    else:
        ticks = None
    cbar = fig.colorbar(cax, format='%.3g', ticks=ticks)


makePlot(None, 'Regular linear scale')

# Example of logarithm normalization using FuncNorm
norm = colors.FuncNorm(f=lambda x: np.log10(x),
                       finv=lambda x: 10.**(x), vmin=0.01, vmax=2)
makePlot(norm, "Log normalization using FuncNorm")
# The same can be achived with
# norm = colors.FuncNorm(f='log',vmin=0.01,vmax=2)

# Example of root normalization using FuncNorm
norm = colors.FuncNorm(f='sqrt', vmin=0.0, vmax=2)
makePlot(norm, "Root normalization using FuncNorm")

# Performing a symmetric amplification of the features around 0
norm = colors.MirrorPiecewiseNorm(fpos='crt')
makePlot(norm, "Amplified features symetrically around \n"
               "0 with MirrorPiecewiseNorm")


# Amplifying features near 0.6 with MirrorPiecewiseNorm
norm = colors.MirrorPiecewiseNorm(fpos='crt', fneg='crt',
                                  center_cm=0.35,
                                  center_data=0.6)
makePlot(norm, "Amplifying positive and negative features\n"
               "standing on 0.6 with MirrorPiecewiseNorm")

# Amplifying features near both -0.4 and near 1.2 with PiecewiseNorm
norm = colors.PiecewiseNorm(flist=['cubic', 'crt', 'cubic', 'crt'],
                            refpoints_cm=[0.25, 0.5, 0.75],
                            refpoints_data=[-0.4, 1, 1.2])
makePlot(norm, "Amplifying positive and negative features standing\n"
               " on -0.4 and on 1.2 with PiecewiseNorm")

# Amplifying features near both -0.4 and near 1.2 with PiecewiseNorm
norm = colors.PiecewiseNorm(flist=['linear', 'crt', 'crt'],
                            refpoints_cm=[0.2, 0.6],
                            refpoints_data=[-0.6, 1.2])
makePlot(norm, "Amplifying only positive features standing on -0.6\n"
               " and on 1.2 with PiecewiseNorm")


plt.show()
