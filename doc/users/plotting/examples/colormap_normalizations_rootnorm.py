"""
Demonstration of using the RootNorm classes for normalization.
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

xmax = 16 * np.pi
x = np.linspace(0, xmax, 1024)
y = np.linspace(-2, 1, 512)
X, Y = np.meshgrid(x, y)

data = np.zeros(X.shape)


def gauss2d(x, y, a0, x0, y0, wx, wy):
    return a0 * np.exp(-(x - x0)**2 / wx**2 - (y - y0)**2 / wy**2)

N = 61
for i in range(N):
    data = data + gauss2d(X, Y, 2. * i / N, i *
                          (xmax / N), -0.25, xmax / (3 * N), 0.07)
    data = data - gauss2d(X, Y, 1. * i / N, i *
                          (xmax / N), -0.75, xmax / (3 * N), 0.07)

data[Y > 0] = np.cos(X[Y > 0]) * Y[Y > 0]**2

N = 61
for i, val in enumerate(np.linspace(-1, 1, N)):
    if val < 0:
        aux = val
    if val > 0:
        aux = val * 2
    data[(X > (i * (xmax / N))) * (Y < -1)] = aux

cmap = cm.gist_rainbow

norms = [('Linear Scale', None),
         ('Symmetric root norm',
          colors.SymRootNorm(orderpos=7, orderneg=2, center=0.3)),
         ('Positive root norm',
          colors.PositiveRootNorm(vmin=0, orderpos=5)),
         ('Negative root norm',
          colors.NegativeRootNorm(vmax=0, orderneg=5))]


for label, norm in norms:
    fig, ax = plt.subplots()
    cax = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm)
    ax.set_title(label)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-2, 1)
    if norm:
        ticks = norm.ticks()
    else:
        ticks = None
    cbar = fig.colorbar(cax, format='%.3g', ticks=ticks)

plt.show()
