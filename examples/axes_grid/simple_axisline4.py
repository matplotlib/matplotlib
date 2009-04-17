import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import numpy as np

fig = plt.figure(1, (4,3))

ax = SubplotHost(fig, 111)
fig.add_subplot(ax)

xx = np.arange(0, 2*np.pi, 0.01)
ax.plot(xx, np.sin(xx))

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels(["0", r"$\frac{1}{2}\pi$",
                     r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

ax2.axis["right"].major_ticklabels.set_visible(False)

plt.draw()
plt.show()

