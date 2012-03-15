from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# wireframe
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x,y,z, rstride=4, cstride=4, color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# sub-sample data-set
x = x[::2, ::2]
y = y[::2, ::2]
z = z[::2, ::2]

# lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot3D requires a 1D array for x, y, and z np.ravel() flattens it's
# input
ax.plot3D(np.ravel(x),np.ravel(y),np.ravel(z), color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# scatter3D requires a 1D array for x, y, and z np.ravel() flattens
# it's input
ax.scatter3D(np.ravel(x),np.ravel(y),np.ravel(z), color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
