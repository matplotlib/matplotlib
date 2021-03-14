"""
===================================
pcolormesh([X, Y], Z, [cmap=], ...)
===================================

`~.axes.Axes.pcolormesh` is more flexible than `~.axes.Axes.imshow` in that
the x and y vectors need not be equally spaced (indeed they can be skewed).

"""
import matplotlib.pyplot as plt
import numpy as np

# make data:
np.random.seed(3)
Z = np.random.uniform(0.1, 1.0, (8, 8))
x = np.array([0, 0.5, 1.5, 3, 5.2, 6.3, 7.2, 7.5, 8])
y = 1.2**np.arange(0, 9)

# plot
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    # plot:
    ax.pcolormesh(x, y, Z, cmap=plt.get_cmap('Oranges'), vmin=0, vmax=1.1)

    ax.set_ylim(1, np.max(y))
    ax.set_xlim(np.min(x), np.max(x))
    plt.show()
