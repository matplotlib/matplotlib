#!/usr/bin/env python
"""
Use a pcolor or imshow with a custom colormap to make a contour plot.

Since this example was initially written, a proper contour routine was
added to matplotlib - see contour_demo.py and
http://matplotlib.sf.net/matplotlib.pylab.html#-contour.
"""

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
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    return 1.0/(2*pi*sigmax*sigmay*(1-rho**2)) * exp( -z/(2*(1-rho**2)))


delta = 0.01
x = arange(-3.0, 3.0, delta)
y = arange(-3.0, 3.0, delta)
X,Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2 - Z1 # difference of Gaussians

cmap = cm.get_cmap('jet', 10)    # 10 discrete colors

im = imshow(Z, cmap=cmap, interpolation='bilinear')
axis('off')
colorbar(tickfmt='%1.2f')
clim(-.1, .1)
#savefig('test')
show()

