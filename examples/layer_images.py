#!/usr/bin/env python
"""
See pcolor_demo2 for a much faster way of generating pcolor plots
"""
from __future__ import division
from matplotlib.matlab import *

def func3(x,y):
    return (1- x/2 + x**5 + y**3)*exp(-x**2-y**2)


# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = arange(-3.0, 3.0, dx)
y = arange(-3.0, 3.0, dy)
X,Y = meshgrid(x, y)

Z1 = rand(10,6)
im1 = imshow(Z1, cmap=cm.gray)
im1.set_interpolation('nearest')
hold(True)

Z2 = func3(X, Y)
im2 = imshow(Z2, cmap=cm.jet, alpha=0.75)
im2.set_interpolation('nearest')


show()

    
