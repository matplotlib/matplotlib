from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(0, 2*np.pi)
x = np.cos(theta)
y = np.sin(theta)
z = theta
markerline, stemlines, baseline = ax.stem(x, y, z)

plt.show()
