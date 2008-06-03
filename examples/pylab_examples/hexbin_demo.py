'''
hexbin is an axes method or pyplot function that is essentially
a pcolor of a 2-D histogram with hexagonal cells.  It can be
much more informative than a scatter plot; in the first subplot
below, try substituting 'scatter' for 'hexbin'.
'''

from matplotlib.pyplot import *
import numpy as np

n = 100000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

subplot(121)
hexbin(x,y)
axis([xmin, xmax, ymin, ymax])
title("Hexagon binning")
cb = colorbar()
cb.set_label('counts')

subplot(122)
hexbin(x,y,bins='log')
axis([xmin, xmax, ymin, ymax])
title("With a log color scale")
cb = colorbar()
cb.set_label('log10(N)')

show()

