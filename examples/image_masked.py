#!/usr/bin/env python
'''imshow with masked array input and out-of-range colors.

'''

from pylab import *
import matplotlib.numerix.npyma as ma
import matplotlib.colors as colors

delta = 0.025
x = y = arange(-3.0, 3.0, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z2-Z1)  # difference of Gaussians

# Set up a colormap:
palette = cm.gray
palette.set_over('r', 1.0)
palette.set_under('g', 1.0)
palette.set_bad('b', 1.0)
# Alternatively, we could use
# palette.set_bad(alpha = 0.0)
# to make the bad region transparent.  This is the default.
# If you comment out all the palette.set* lines, you will see
# all the defaults; under and over will be colored with the
# first and last colors in the palette, respectively.
Zm = ma.masked_where(Z > 1.2, Z)

# By setting vmin and vmax in the norm, we establish the
# range to which the regular palette color scale is applied.
# Anything above that range is colored based on palette.set_over, etc.

im = imshow(Zm, interpolation='bilinear',
    cmap=palette,
    norm = colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False),
    origin='lower', extent=[-3,3,-3,3])
title('Green=low, Red=high, Blue=bad')
colorbar(im, extend='both', shrink=0.8)
show()
