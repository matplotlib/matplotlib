import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['toolbar'] = 'toolmanager'

X, Y, Z = axes3d.get_test_data(0.05)
dx = X[0, 1] - X[0, 0]
dy = Y[1, 0] - Y[0, 0]
extent = (X[0, 0], X[0, -1] + dx, Y[0, 0], Y[-1, 0] + dy)

fig = plt.figure()

z = np.linspace(np.min(Z), np.max(Z))
dat = Z-z[0]

im = plt.imshow(Z-z[0], cmap='coolwarm', extent=extent, vmin=np.min(Z),
                vmax=np.max(Z))


def updatefig(i):
    im.set_array(Z - z[i])
    return im,

slider = animation.AnimationSlider(fig, z, 'z=', orientation='vertical')
ani = animation.FuncAnimation(fig, updatefig, slider.generator, interval=50,
                              blit=True)
plt.show()
