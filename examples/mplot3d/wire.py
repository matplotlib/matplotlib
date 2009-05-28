from mpl_toolkits.mplot3d import axes3d
import pylab
import random
import numpy as np

fig = pylab.figure()
ax = axes3d.Axes3D(fig)
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

pylab.show()

