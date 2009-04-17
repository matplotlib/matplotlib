import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
from demo_image import get_demo_image

F = plt.figure(1, (5.5, 3.5))
grid = AxesGrid(F, 111, # similar to subplot(111)
                nrows_ncols = (1, 3),
                axes_pad = 0.1,
                add_all=True,
                label_mode = "L",
                )

Z, extent = get_demo_image() # demo image

im1=Z
im2=Z[:,:10]
im3=Z[:,10:]
vmin, vmax = Z.min(), Z.max()
for i, im in enumerate([im1, im2, im3]):
    ax = grid[i]
    ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

plt.draw()
plt.show()
