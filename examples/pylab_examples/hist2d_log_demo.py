from matplotlib.colors import LogNorm
from pylab import *

#normal distribution center at x=0 and y=5
x = randn(100000)
y = randn(100000)+5

hist2d(x, y, bins=40, norm=LogNorm())
colorbar()
show()
