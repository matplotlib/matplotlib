from mpl_toolkits.mplot3d import Axes3D
import pylab
import random

fig = pylab.figure()
ax = Axes3D(fig)
n = 100
for c, zl, zh in [('r', -50, -25), ('b', -30, -5)]:
    xs, ys, zs = zip(*
                   [(random.randrange(23, 32),
                     random.randrange(100),
                     random.randrange(zl, zh)
                     ) for i in range(n)])
    ax.scatter(xs, ys, zs, c=c)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

pylab.show()

