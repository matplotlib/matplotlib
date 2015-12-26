import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axes_divider import make_axes_locatable

from mpl_toolkits.axes_grid.colorbar import colorbar
# from matplotlib.pyplot import colorbar

fig = plt.figure(1, figsize=(6, 3))
fig.subplots_adjust(wspace=0.5)

ax1 = fig.add_subplot(121)
im1 = ax1.imshow([[1,2],[3,4]])

ax1_divider = make_axes_locatable(ax1)
cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
cb1 = colorbar(im1, cax=cax1)

ax2 = fig.add_subplot(122)
im2 = ax2.imshow([[1,2],[3,4]])

ax2_divider = make_axes_locatable(ax2)
cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
cb2 = colorbar(im2, cax=cax2, orientation="horizontal")
cax2.xaxis.set_ticks_position("top")
plt.show()

