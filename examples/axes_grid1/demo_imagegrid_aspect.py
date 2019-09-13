"""
=====================
Demo Imagegrid Aspect
=====================

"""
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure()

grid1 = ImageGrid(fig, 121, (2, 2), axes_pad=0.1,
                  aspect=True, share_all=True)

for i in [0, 1]:
    grid1[i].set_aspect(2)


grid2 = ImageGrid(fig, 122, (2, 2), axes_pad=0.1,
                  aspect=True, share_all=True)


for i in [1, 3]:
    grid2[i].set_aspect(2)

plt.show()
