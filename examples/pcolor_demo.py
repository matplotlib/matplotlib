from __future__ import division
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
#axis([-3, 3, -3, 3])
#axis('off')
#savefig('pcolor_demo')
show()

    
