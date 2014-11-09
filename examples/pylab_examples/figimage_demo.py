"""
This illustrates placing images directly in the figure, with no axes.

"""
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


fig = plt.figure()
Z = np.arange(10000.0)
Z.shape = 100, 100
Z[:, 50:] = 1.

im1 = plt.figimage(Z, xo=50, yo=0, cmap=cm.jet, origin='lower')
im2 = plt.figimage(Z, xo=100, yo=100, alpha=.8, cmap=cm.jet, origin='lower')

plt.show()
