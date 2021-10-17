"""
==================================
Colormap Normalizations SymLogNorm
==================================

Demonstration of using norm to map colormaps onto data in non-linear ways.

.. redirect-from:: /gallery/userdemo/colormap_normalization_symlognorm
"""

########################################
# Synthetic dataset consisting of two humps, one negative and one positive,
# the positive with 8-times the amplitude.
# Linearly, the negative hump is almost invisible,
# and it is very difficult to see any detail of its profile.
# With the logarithmic scaling applied to both positive and negative values,
# it is much easier to see the shape of each hump.
#
# See `~.colors.SymLogNorm`.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

N = 200
gain = 8
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
rbf = lambda x, y: 1.0 / (1 + 5 * ((x ** 2) + (y ** 2)))
Z1 = rbf(X + 0.5, Y + 0.5)
Z2 = rbf(X - 0.5, Y - 0.5)
Z = gain * Z1 - Z2

shadeopts = { 'cmap': 'PRGn', 'shading': 'gouraud' }
colormap = 'PRGn'
lnrwidth = 0.2

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=lnrwidth, linscale=1,
                                              vmin=-gain, vmax=gain, base=10),
                       **shadeopts)
fig.colorbar(pcm, ax=ax[0], extend='both')
ax[0].text(-2.5, 1.5, 'symlog')

pcm = ax[1].pcolormesh(X, Y, Z, vmin=-gain, vmax=gain,
                       **shadeopts)
fig.colorbar(pcm, ax=ax[1], extend='both')
ax[1].text(-2.5, 1.5, 'linear')


########################################
# Clearly, tt may be necessary to experiment with multiple different
# colorscales in order to find the best visualization for
# any particular dataset.
# As well as the `~.colors.SymLogNorm` scaling, there is also
# the option of using the `~.colors.AsinhNorm`, which has a smoother
# transition between the linear and logarithmic regions of the transformation
# applied to the "z" axis.
# In the plots below, it may be possible to see ring-like artifacts
# in the lower-amplitude, negative, hump shown in purple despite
# there being no sharp features in the dataset itself.
# The ``asinh`` scaling shows a smoother shading of each hump.

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=lnrwidth, linscale=1,
                                              vmin=-gain, vmax=gain, base=10),
                       **shadeopts)
fig.colorbar(pcm, ax=ax[0], extend='both')
ax[0].text(-2.5, 1.5, 'symlog')

pcm = ax[1].pcolormesh(X, Y, Z,
                       norm=colors.AsinhNorm(linear_width=lnrwidth,
                                             vmin=-gain, vmax=gain),
                       **shadeopts)
fig.colorbar(pcm, ax=ax[1], extend='both')
ax[1].text(-2.5, 1.5, 'asinh')


plt.show()
