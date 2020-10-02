"""
===========================
Demo Colorbar of Inset Axes
===========================

"""

from matplotlib import cbook
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes


fig, ax = plt.subplots(figsize=[5, 4])

Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)
extent = (-3, 4, -4, 3)

ax.set(aspect=1, xlim=(-15, 15), ylim=(-20, 5))

axins = zoomed_inset_axes(ax, zoom=2, loc='upper left')
im = axins.imshow(Z, extent=extent, origin="lower")

plt.xticks(visible=False)
plt.yticks(visible=False)

# colorbar
cax = inset_axes(axins,
                 width="5%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axins.transAxes,
                 borderpad=0,
                 )
fig.colorbar(im, cax=cax)

plt.show()
