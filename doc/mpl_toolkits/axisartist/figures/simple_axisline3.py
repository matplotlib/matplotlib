import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot

fig = plt.figure(1, (3,3))

ax = Subplot(fig, 111)
fig.add_subplot(ax)

ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)

plt.show()
