#!/usr/bin/env python
"""
See pcolor_demo2 for an alternative way of generating pcolor plots
using imshow that is likely faster for large grids
"""
from __future__ import division
from matplotlib.patches import Patch
from matplotlib.matlab import *

def func3(x,y):
    return (1- x/2 + x**5 + y**3)*exp(-x**2-y**2)


# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = arange(-3.0, 3.0, dx)
y = arange(-3.0, 3.0, dy)
X,Y = meshgrid(x, y)

Z = func3(X, Y)
pcolor(X, Y, Z, shading='flat')
colorbar()
savefig('pcolor_demo')
show()

    
