from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(-np.pi/2, np.pi/2, 40)
y = [1]*len(x)
z = np.cos(x)
markerline, stemlines, baseline = ax.stem(x, y, z, '-.', zdir='-y')
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(baseline, 'color', 'r', 'linewidth', 1)

plt.show()
