"""
You can specify whether images should be plotted with the array origin
x[0,0] in the upper left or upper right by using the origin parameter.
You can also control the default be setting image.origin in your
matplotlibrc file; see http://matplotlib.org/matplotlibrc
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(120)
x.shape = (10, 12)

interp = 'bilinear'
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))
axs[0].set_title('blue should be up')
axs[0].imshow(x, origin='upper', interpolation=interp)

axs[1].set_title('blue should be down')
axs[1].imshow(x, origin='lower', interpolation=interp)
plt.show()
