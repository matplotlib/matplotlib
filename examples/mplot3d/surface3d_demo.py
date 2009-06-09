from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import random
import numpy as np

fig = pylab.figure()
ax = Axes3D(fig)
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)

pylab.show()

