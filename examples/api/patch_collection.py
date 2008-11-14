import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import pylab

fig=pylab.figure()
ax=fig.add_subplot(111)

resolution = 50 # the number of vertices
N = 3
x       = pylab.rand(N)
y       = pylab.rand(N)
radii   = 0.1*pylab.rand(N)
patches = []
for x1,y1,r in zip(x, y, radii):
    circle = Circle((x1,y1), r)
    patches.append(circle)

x       = pylab.rand(N)
y       = pylab.rand(N)
radii   = 0.1*pylab.rand(N)
theta1  = 360.0*pylab.rand(N)
theta2  = 360.0*pylab.rand(N)
for x1,y1,r,t1,t2 in zip(x, y, radii, theta1, theta2):
    wedge = Wedge((x1,y1), r, t1, t2)
    patches.append(wedge)

# Some limiting conditions on Wedge
patches += [
    Wedge((.3,.7), .1, 0, 360),             # Full circle
    Wedge((.7,.8), .2, 0, 360, width=0.05), # Full ring
    Wedge((.8,.3), .2, 0, 45),              # Full sector
    Wedge((.8,.3), .2, 45, 90, width=0.10), # Ring sector
]

for i in range(N):
    polygon = Polygon(pylab.rand(N,2), True)
    patches.append(polygon)

colors = 100*pylab.rand(len(patches))
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
p.set_array(pylab.array(colors))
ax.add_collection(p)
pylab.colorbar(p)

pylab.show()
