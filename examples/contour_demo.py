#!/usr/bin/env python
'''
Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also contour_image.py.
'''
from pylab import *

import sys
fignum = 0
if len(sys.argv) == 2:
    fignum = int(sys.argv[1])

delta = 0.025
x = arange(-3.0, 3.0, delta)
y = arange(-2.0, 2.0, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)



# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
figure()
levels, colls = contour(X, Y, Z)
clabel(colls, inline=1, fontsize=10)
title('Simplest default with labels')


# You can force all the contours to be the same color
figure()
levels, colls = contour(X, Y, Z, 6,
                        colors=('k',)
                        )
clabel(colls, levels, fontsize=9, inline=1)
title('Single color')


# And you can manually specify the colors of the contour
figure()
levels, colls = contour(X, Y, Z, 6,
                        linewidths=arange(.5, 4, .5),
                        colors=('r', 'green', 'blue', (1,1,0), '#afeeee', 0.5)
                        )
clabel(colls, levels, fontsize=9, inline=1)
title('Crazy lines')


# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines
figure()
im = imshow(Z, interpolation='bilinear', origin='lower',
            cmap=cm.gray, extent=(-3,3,-2,2))
levels, colls = contour(Z, arange(-1.2,1.6,0.2),
                        origin='lower',
                        linewidths=2,
                        extent=(-3,3,-2,2))

clabel(colls, levels,
       inline=1,
       fmt='%1.1f',
       fontsize=10)

colorbar()  # make a colorbar for the contour lines
title('Lines with colorbar')
hot()  # Now change the colormap for the contour lines and colorbar


#savefig('contour_demo')
show()
