#!/usr/bin/env python
from pylab import *

delta = 0.025
x = y = arange(-3.0, 3.0, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)

# difference of Gaussians
im = imshow(Z2-Z1, interpolation='bilinear', cmap=cm.gray, extent=(-3,3,-3,3))
levels, colls = contour(X, Y, Z2-Z1, linewidths=2)
colorbar()
hot()
savefig('test')
show()

