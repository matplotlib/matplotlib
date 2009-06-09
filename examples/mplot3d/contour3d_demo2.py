from mpl_toolkits.mplot3d import axes3d
import pylab
import random

fig = pylab.figure()
ax = axes3d.Axes3D(fig)
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, 16, extend3d=True)
ax.clabel(cset, fontsize=9, inline=1)

pylab.show()

