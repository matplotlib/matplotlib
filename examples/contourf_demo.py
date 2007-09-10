#!/usr/bin/env python
from pylab import *
import matplotlib.numerix.npyma as ma
origin = 'lower'
#origin = 'upper'

test_masking = False  # There is a bug in filled contour masking.

if test_masking:
    # Use a coarse grid so only a few masked points are needed.
    delta = 0.5
else:
    delta = 0.025

x = y = arange(-3.0, 3.01, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z1 - Z2)

# interior badmask doesn't work yet for filled contours
if test_masking:
    badmask = zeros(shape(Z))

    badmask[5,5] = 1
    badmask[5,6] = 1
    Z[5,5] = 0
    Z[5,6] = 0

    badmask[0,0] = 1
    Z[0,0] = 0
    Z = ma.array(Z, mask=badmask)

# We are using automatic selection of contour levels;
# this is usually not such a good idea, because they don't
# occur on nice boundaries, but we do it here for purposes
# of illustration.
CS = contourf(X, Y, Z, 10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cm.bone,
                        origin=origin)

# Note that in the following, we explicitly pass in a subset of
# the contour levels used for the filled contours.  Alternatively,
# We could pass in additional levels to provide extra resolution.

CS2 = contour(X, Y, Z, CS.levels[::2],
                        colors = 'r',
                        origin=origin,
                        hold='on')

title('Nonsense')
xlabel('word length anomaly')
ylabel('sentence length anomaly')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
# Add the contour line levels to the colorbar
cbar.add_lines(CS2)

figure()

# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.
levels = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5]
CS3 = contourf(X, Y, Z, levels,
                        colors = ('r', 'g', 'b'),
                        origin=origin)

CS4 = contour(X, Y, Z, levels,
                       colors = ('k',),
                       linewidths = (3,),
                       origin = origin)
title('Listed colors')
clabel(CS4, fmt = '%2.1f', colors = 'w', fontsize=14)
colorbar(CS3)

#savefig('contourf_demo')
show()

