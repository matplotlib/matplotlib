"""
===================================
pcolormesh([X, Y], Z, [cmap=], ...)
===================================

`~.axes.Axes.pcolormesh` is more flexible than `~.axes.Axes.imshow` in that
the x and y vectors need not be equally spaced (indeed they can be skewed).

"""
import matplotlib.pyplot as plt
import numpy as np

# make data
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
Z = (1 - X/2. + X**5 + Y**3) * np.exp(-X**2 - Y**2)
Z = Z - Z.min()

# plot
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    # plot:
    ax.pcolormesh(X, Y, Z, vmin=0, vmax=1.1)

    #ax.set_ylim(np.min(Y), np.max(Y))
    #ax.set_xlim(np.min(X), np.max(X))
    plt.show()
