#!/usr/bin/env python
from pylab import *

delta = 0.025
x = y = arange(-3.0, 3.0, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

im = imshow(Z, interpolation='bilinear', cmap=cm.gray,
            origin='lower', extent=[-3,3,-3,3])

#savefig('image_demo')
show()

