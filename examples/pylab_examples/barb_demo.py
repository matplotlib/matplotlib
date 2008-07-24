'''
Demonstration of wind barb plots
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 5)
X,Y = np.meshgrid(x, x)
U, V = 12*X, 12*Y

data = [(-1.5,.5,-6,-6),
        (1,-1,-46,46),
        (-3,-1,11,-11),
        (1,1.5,80,80)]

#Default parameters for arbitrary set of vectors
ax = plt.subplot(2,2,1)
ax.barbs(*zip(*data))

#Default parameters, uniform grid
ax = plt.subplot(2,2,2)
ax.barbs(X, Y, U, V)

#Change parameters for arbitrary set of vectors
ax = plt.subplot(2,2,3)
ax.barbs(flagcolor='r', barbcolor=['b','g'], barb_increments=dict(half=10,
    full=20, flag=100), *zip(*data))

#Showing colormapping with uniform grid. 
ax = plt.subplot(2,2,4)
ax.barbs(X, Y, U, V, np.sqrt(U*U + V*V), fill_empty=True, rounding=False)

plt.show()
