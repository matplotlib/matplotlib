from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import pylab
import random
import numpy as np

fig = pylab.figure()
ax = Axes3D(fig)
x = np.random.rand(100) * 4
y = np.random.rand(100) * 4
hist, xedges, yedges = np.histogram2d(x, y, bins=4)

elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = np.meshgrid(
        [xedges[i] + 0.25 for i in range(len(xedges) - 1)],
        [yedges[i] + 0.25 for i in range(len(yedges) - 1)])
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = [0] * elements
dx = [0.5] * elements
dy = [0.5] * elements
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')

pylab.show()

