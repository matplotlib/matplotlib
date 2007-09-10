from matplotlib.pyplot import figure, show
import numpy as npy
from matplotlib.image import NonUniformImage

x = npy.arange(-4, 4, 0.005)
y = npy.arange(-4, 4, 0.005)
print 'Size %d points' % (len(x) * len(y))
z = npy.sqrt(x[npy.newaxis,:]**2 + y[:,npy.newaxis]**2)

fig = figure()
ax = fig.add_subplot(111)
im = NonUniformImage(ax, extent=(-4,4,-4,4))
im.set_data(x, y, z)
ax.images.append(im)
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)

fig2 = figure()
ax = fig2.add_subplot(111)
x2 = x**3
im = NonUniformImage(ax, extent=(-64,64,-4,4))
im.set_data(x2, y, z)
ax.images.append(im)
ax.set_xlim(-64,64)
ax.set_ylim(-4,4)
show()
