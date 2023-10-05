# importing numpy package
import numpy as np
# importing matplotlib package
import matplotlib.pyplot as plt
 
# Creating an empty canvas(figure)
fig = plt.figure()
 
# Using the gca function, we are defining
# the current axes as a 3D projection
ax = fig.gca(projection='3d')
 
# Labelling X-Axis
ax.set_xlabel('X-Axis')
 
# Labelling Y-Axis
ax.set_ylabel('Y-Axis')
 
# Labelling Z-Axis
ax.set_zlabel('Z-Axis')
 
# Creating 10 values for X
x = [1,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]
 
# Creating 10 values for Y
y = [1,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]
 
# Creating 10 random values for Y
z=[1,2,4,5,6,7,8,9,10,11]
 
# zdir='z' fixes all the points to zs=0 and
# (x,y) points are plotted in the x-y axis
# of the graph
ax.plot(x, y, zs=0, zdir='z')
 
# zdir='y' fixes all the points to zs=0 and
# (x,y) points are plotted in the x-z axis of the
# graph
ax.plot(x, y, zs=0, zdir='y')
 
# Showing the above plot
plt.show()
