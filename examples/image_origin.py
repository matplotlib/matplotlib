"""
You can specify whether images should be plotted with the array origin
x[0,0] in the upper left or upper right by using the origin parameter.
You can also control the default be setting image.origin in your
matplotlibrc file; see http://matplotlib.sourceforge.net/.matplotlibrc
"""
from matplotlib.matlab import *

x = arange(100.0); x.shape = 10,10


subplot(211)
title('blue should be up')
imshow(x, origin='upper', interpolation='nearest')

subplot(212)
title('blue should be down')
imshow(x, origin='lower', interpolation='nearest')

savefig('image_origin')
show()
