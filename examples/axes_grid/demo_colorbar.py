import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.colorbar import outer_colorbar, inner_colorbar


plt.figure(figsize=[6,6])
gs = gridspec.GridSpec(3,3, hspace=0.7, wspace=0.7)

locs = [1, 2, 3, 4, 6, 7, 8, 9, 10]
arr = np.arange(100).reshape((10,10))
for i, ss in zip(locs, gs):
    ax = plt.subplot(ss)
    ax.axison = False
    im = ax.imshow(arr, interpolation="none")
    cb = outer_colorbar(ax, mappable=im, loc=i,
                        orientation=None,
                        ticks=[0, 50],
                        length="50%", thickness="7%",
                        pad=0.2,
                        )




plt.figure(figsize=[6,6])
gs = gridspec.GridSpec(3,3)

from matplotlib.patheffects import withStroke
pe = [withStroke(foreground="w", linewidth=3)]

locs = [1, 2, 3, 4, 6, 7, 8, 9, 10]
arr = np.arange(100).reshape((10,10))
for i, ss in zip(locs, gs):
    ax = plt.subplot(ss)
    ax.axison = False
    im = ax.imshow(arr, interpolation="none")
    cb = inner_colorbar(ax, mappable=im, loc=i,
                        orientation=None,
                        ticks=[0, 50],
                        length="50%", thickness="5%",
                        pad=0.2,
                        )
    cb.ax.axis[:].major_ticklabels.set_path_effects(pe)
    

plt.show()
