import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import TICKRIGHT

rx, ry = 3., 1.
area = rx * ry * np.pi
angles = np.linspace(0., 360., 30.)

x, y, sizes, colors = np.random.rand(4, 30)
sizes *= 20**2.

plt.scatter(x, y, sizes, colors, marker="o",zorder=2)
plt.scatter(x, y, 2.5*sizes, colors, marker=TICKRIGHT, angles=angles, zorder=2)

plt.show()
