import matplotlib.pyplot as plt

plt.rcParams["text.usetex"]=False

fig = plt.figure(1, figsize=(6, 3))

ax1 = fig.add_subplot(121)
im1 = ax1.imshow([[1,2],[3,4]])
cb1 = plt.colorbar(im1)
cb1.ax.set_yticks([1, 3])
ax1.set_title("Original MPL's colorbar w/\nset_yticks([1,3])", size=10)

from mpl_toolkits.axes_grid.colorbar import colorbar
ax2 = fig.add_subplot(122)
im2 = ax2.imshow([[1,2],[3,4]])
cb2 = colorbar(im2)
cb2.ax.set_yticks([1, 3])
ax2.set_title("AxesGrid's colorbar w/\nset_yticks([1,3])", size=10)

plt.show()

