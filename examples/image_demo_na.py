#!/usr/bin/env python
from matplotlib import rcParams
rcParams['numerix'] = 'numarray'

from pylab import *


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate gaussan distribution for equal shape X, Y

    http://mathworld.wolfram.com/BivariateNormalDistribution.html
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = (1.0/sigmax**2)*Xmu**2 + (1.0/sigmay)*Ymu**2 - (2*rho/(sigmax*sigmay))*Xmu*Ymu
    return 1.0/(2*pi*sigmax*sigmay*(1-rho**2)) * exp( -1/(2*(1-rho**2))*z)


delta = 0.025
x = arange(-3.0, 3.0, delta)
y = arange(-3.0, 3.0, delta)
X,Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)

# difference of Gaussians
im = imshow(Z2-Z1)

# set the interpolation method: 'nearest', 'bilinear', 'bicubic' and much more
im.set_interpolation('bilinear')


axis('off')
#savefig('test')
show()

