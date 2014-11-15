'''
This illustrates the NonUniformImage class, which still needs
an axes method interface; either a separate interface, or a
generalization of imshow.
'''

from matplotlib.pyplot import figure, show
import numpy as np
from matplotlib.image import NonUniformImage
from matplotlib import cm

interp = 'nearest'

x = np.linspace(-4, 4, 9)
x2 = x**3
y = np.linspace(-4, 4, 9)
#print('Size %d points' % (len(x) * len(y)))
z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)

fig = figure()
fig.suptitle('NonUniformImage class')
ax = fig.add_subplot(221)
im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),
                     cmap=cm.Purples)
im.set_data(x, y, z)
ax.images.append(im)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title(interp)

ax = fig.add_subplot(222)
im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),
                     cmap=cm.Purples)
im.set_data(x2, y, z)
ax.images.append(im)
ax.set_xlim(-64, 64)
ax.set_ylim(-4, 4)
ax.set_title(interp)

interp = 'bilinear'

ax = fig.add_subplot(223)
im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),
                     cmap=cm.Purples)
im.set_data(x, y, z)
ax.images.append(im)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title(interp)

ax = fig.add_subplot(224)
im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),
                     cmap=cm.Purples)
im.set_data(x2, y, z)
ax.images.append(im)
ax.set_xlim(-64, 64)
ax.set_ylim(-4, 4)
ax.set_title(interp)

show()
