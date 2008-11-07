#!/usr/bin/env python
# -*- noplot -*-

"""
This example demonstrates how to set a hyperlinks on various kinds of elements.

This currently only works with the SVG backend.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

f = plt.figure()
s = plt.scatter([1,2,3],[4,5,6])
s.set_urls(['http://www.bbc.co.uk/news','http://www.google.com',None])
f.canvas.print_figure('scatter.svg')

f = plt.figure()
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                origin='lower', extent=[-3,3,-3,3])

im.set_url('http://www.google.com')
f.canvas.print_figure('image.svg')

