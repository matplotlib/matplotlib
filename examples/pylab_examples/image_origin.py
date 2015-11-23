"""
You can specify whether images should be plotted with the array origin
x[0,0] in the upper left or upper right by using the origin parameter.
You can also control the default be setting image.origin in your
matplotlibrc file; see http://matplotlib.org/matplotlibrc
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100.0)
x.shape = (10, 10)

interp = 'bilinear'
#interp = 'nearest'
lim = -2, 11, -2, 6
plt.subplot(211, facecolor='g')
plt.title('blue should be up')
plt.imshow(x, origin='upper', interpolation=interp, cmap='jet')
#plt.axis(lim)

plt.subplot(212, facecolor='y')
plt.title('blue should be down')
plt.imshow(x, origin='lower', interpolation=interp, cmap='jet')
#plt.axis(lim)
plt.show()
