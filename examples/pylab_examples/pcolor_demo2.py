#!/usr/bin/env python
"""
See pcolor_demo2 for a much faster way of generating pcolor plots
"""
from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import cm # colormaps
import numpy as np

def func3(x,y):
    return (1- x/2 + x**5 + y**3)*np.exp(-x**2-y**2)


# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X,Y = np.meshgrid(x, y)

Z = func3(X, Y)


fig, ax = plt.subplots()
im = plt.imshow(Z, cmap=cm.RdBu, vmax=abs(Z).max(), vmin=-abs(Z).max())
#im.set_interpolation('nearest')
#im.set_interpolation('bicubic')
im.set_interpolation('bilinear')
#ax.set_image_extent(-3, 3, -3, 3)

plt.show()


