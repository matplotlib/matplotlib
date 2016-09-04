import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import TICKRIGHT

nbpts = 50
rx, ry = 3., 1.
area = rx * ry * np.pi
angles = np.linspace(0., 360., nbpts)

x, y, sizes, colors = np.random.rand(4, nbpts)
sizes *= 2000.

plt.scatter(x, y, sizes, colors, marker="*", angles=angles, zorder=2)
plt.scatter(x, y, 2.5*sizes, colors, marker=TICKRIGHT, angles=angles, zorder=2)

plt.show()

