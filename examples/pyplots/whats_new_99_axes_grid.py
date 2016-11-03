import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

def get_demo_image():
    # prepare image
    delta = 0.5

    extent = (-3,4,-4,3)
    x = np.arange(-3.0, 4.001, delta)
    y = np.arange(-4.0, 3.001, delta)
    X, Y = np.meshgrid(x, y)
    import matplotlib.mlab as mlab
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = (Z1 - Z2) * 10

    return Z, extent



def get_rgb():
    Z, extent = get_demo_image()

    Z[Z<0] = 0.
    Z = Z/Z.max()

    R = Z[:13,:13]
    G = Z[2:,2:]
    B = Z[:13,2:]

    return R, G, B


fig = plt.figure(1)
ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

r, g, b = get_rgb()
kwargs = dict(origin="lower", interpolation="nearest")
ax.imshow_rgb(r, g, b, **kwargs)

ax.RGB.set_xlim(0., 9.5)
ax.RGB.set_ylim(0.9, 10.6)


plt.draw()
plt.show()
