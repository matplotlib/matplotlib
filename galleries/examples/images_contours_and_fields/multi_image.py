"""
=================================
Multiple images with one colorbar
=================================

Use a single colorbar for multiple images.

Currently, a colorbar can only be connected to one image. The connection
guarantees that the data coloring is consistent with the colormap scale
(i.e. the color of value *x* in the colormap is used for coloring a data
value *x* in the image).

If we want one colorbar to be representative for multiple images, we have
to explicitly ensure consistent data coloring by using the same
data-to-color pipeline for all the images. We ensure this by explicitly
creating a `matplotlib.colorizer.Colorizer` object that we pass to all
the image plotting methods.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colorizer as mcolorizer
import matplotlib.colors as mcolors

np.random.seed(19680801)

datasets = [
    (i+1)/10 * np.random.rand(10, 20)
    for i in range(4)
]

fig, axs = plt.subplots(2, 2)
fig.suptitle('Multiple images')

# create a colorizer with a predefined norm to be shared across all images
norm = mcolors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))
colorizer = mcolorizer.Colorizer(norm=norm)

images = []
for ax, data in zip(axs.flat, datasets):
    images.append(ax.imshow(data, colorizer=colorizer))

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

plt.show()

# %%
# The colors are now kept consistent across all images when changing the
# scaling, e.g. through zooming in the colorbar or via the "edit axis,
# curves and images parameters" GUI of the Qt backend. Additionally,
# if the colormap of the colorizer is changed, (e.g. through the "edit
# axis, curves and images parameters" GUI of the Qt backend) this change
# propagates to the other plots and the colorbar.
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colorizer.Colorizer`
#    - `matplotlib.colors.Normalize`
