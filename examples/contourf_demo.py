#!/usr/bin/env python
from pylab import *

origin = 'lower'
#origin = 'upper'

delta = 0.5 #0.025
x = y = arange(-3.0, 3.01, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z1 - Z2

# interior badmask doesn't work yet for filled contours
if 0:
    badmask = zeros(shape(Z))

    badmask[5,5] = 1
    badmask[5,6] = 1
    Z[5,5] = 0
    Z[5,6] = 0

    badmask[0,0] = 1
    Z[0,0] = 0

levels, colls = contourf(X, Y, Z, 10, # [-1, -0.1, 0, 0.1],
#                        badmask=badmask,
                        #alpha=0.5,
                        cmap=cm.bone,
                        origin=origin)

levs2, colls2 = contour(X, Y, Z, levels,
#                        badmask=badmask,
                        colors = 'r',
                        origin=origin,
                        hold='on')
savefig('test')
show()

