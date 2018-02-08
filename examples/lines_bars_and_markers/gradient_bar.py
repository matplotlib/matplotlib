"""
============
Gradient Bar
============

"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)


def gbar(ax, x, y, width=0.5, bottom=0):
    X = [[.6, .6], [.7, .7]]
    for left, top in zip(x, y):
        right = left + width
        ax.imshow(X, interpolation='bicubic', cmap=plt.cm.Blues,
                  extent=(left, right, bottom, top), alpha=1)


xmin, xmax = xlim = 0, 10
ymin, ymax = ylim = 0, 1

fig, ax = plt.subplots()
ax.set(xlim=xlim, ylim=ylim, autoscale_on=False)

X = [[.6, .6], [.7, .7]]
ax.imshow(X, interpolation='bicubic', cmap=plt.cm.copper,
          extent=(xmin, xmax, ymin, ymax), alpha=1)

N = 10
x = np.arange(N) + 0.25
y = np.random.rand(N)
gbar(ax, x, y, width=0.7)
ax.set_aspect('auto')
plt.show()
