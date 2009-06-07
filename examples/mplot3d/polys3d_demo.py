from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import pylab
import random
import numpy as np

fig = pylab.figure()
ax = Axes3D(fig)

cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

xs = np.arange(0, 10, 0.4)
verts = []
zs = [0.0, 1.0, 2.0, 3.0]
for z in zs:
    ys = [random.random() for x in xs]
    ys[0], ys[-1] = 0, 0
    verts.append(zip(xs, ys))

poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
                                           cc('y')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlim3d(0, 10)
ax.set_ylim3d(-1, 4)
ax.set_zlim3d(0, 1)

pylab.show()

