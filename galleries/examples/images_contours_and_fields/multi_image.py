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
to explicitly ensure consistent data coloring by using the same data
normalization for all the images. We ensure this by explicitly creating a
``norm`` object that we pass to all the image plotting methods.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors

np.random.seed(19680801)

datasets = [
    (i+1)/10 * np.random.rand(10, 20)
    for i in range(4)
]

fig, axs = plt.subplots(2, 2)
fig.suptitle('Multiple images')

# create a single norm to be shared across all images
norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))

images = []
for ax, data in zip(axs.flat, datasets):
    images.append(ax.imshow(data, norm=norm))

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

plt.show()

# %%
# The colors are now kept consistent across all images when changing the
# scaling, e.g. through zooming in the colorbar or via the "edit axis,
# curves and images parameters" GUI of the Qt backend. This is sufficient
# for most practical use cases.
#
# Advanced: Additionally sync the colormap
# ----------------------------------------
#
# Sharing a common norm object guarantees synchronized scaling because scale
# changes modify the norm object in-place and thus propagate to all images
# that use this norm. This approach does not help with synchronizing colormaps
# because changing the colormap of an image (e.g. through the "edit axis,
# curves and images parameters" GUI of the Qt backend) results in the image
# referencing the new colormap object. Thus, the other images are not updated.
#
# To update the other images, sync the
# colormaps using the following code::
#
#     def sync_cmaps(changed_image):
#         for im in images:
#         if changed_image.get_cmap() != im.get_cmap():
#             im.set_cmap(changed_image.get_cmap())
#
#     for im in images:
#         im.callbacks.connect('changed', sync_cmaps)
#
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Normalize`
#    - `matplotlib.cm.ScalarMappable.set_cmap`
#    - `matplotlib.cbook.CallbackRegistry.connect`
