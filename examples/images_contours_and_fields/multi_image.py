"""
==================================
Multiple images sharing a colorbar
==================================

A colorbar is always connected to a single image (more precisely, to a single
`.ScalarMappable`). If we want to use multiple images with a single colorbar,
we have to create a colorbar for one of the images and make sure that the
other images use the same color mapping, i.e. the same Colormap and the same
`.Normalize` instance.

"""
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
nrows = 3
ncols = 2
cmap = "plasma"

data = [((1 + i) / 10) * np.random.rand(11, 21) * 1e-6
        for i in range(nrows * ncols)]

#############################################################################
#
# All data available beforehand
# -----------------------------
#
# In the most simple case, we have all the image data sets available, e.g.
# in a list, before we start to plot. In this case, we create a common
# `.Normalize` instance scaling to the global min and max by passing all data
# to the *data* argument of Normalize. We then use this and a common colormap
# when creating the images.

fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
fig.suptitle('Multiple images sharing a colorbar')

norm = colors.Normalize(data=data)
images = [ax.imshow(data, cmap=cmap, norm=norm)
          for data, ax in zip(data, axs.flat)]
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.06)
plt.show()

#############################################################################
#
# Not all data available beforehand
# ---------------------------------
#
# Things get a bit more complicated, if we don't have all the data beforehand,
# e.g. when we generate or load the data just before each plot command. In
# this case, the common norm has to be created and set afterwards. We can use
# a small helper function for that.


def normalize_images(images):
    """Normalize the given images to their global min and max."""
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)


fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
fig.suptitle('Multiple images sharing a colorbar')

images = []
for i in range(nrows):
    for j in range(ncols):
        # Generate data with a range that varies from one plot to the next.
        data = ((1 + i + j) / 10) * np.random.rand(11, 21) * 1e-6
        images.append(axs[i, j].imshow(data, cmap=cmap))
normalize_images(images)
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.06)


#############################################################################
#
# Dynamically adapting to changes of the norm and cmap
# ----------------------------------------------------
#
# If the images norm or cmap can change later on (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), one can propagate
# these changes to all images by connecting to the 'changed' callback.
#
# Note: It's important to have the ``if`` statement to check whether there
# are really changes to apply. Otherwise, you would run into an infinite
# recursion with all images notifying each other infinitely.

def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacksSM.connect('changed', update)


#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods and classes is shown
# in this example:

import matplotlib
matplotlib.axes.Axes.imshow
matplotlib.pyplot.imshow
matplotlib.figure.Figure.colorbar
matplotlib.pyplot.colorbar
matplotlib.colors.Normalize
matplotlib.cm.ScalarMappable.set_cmap
matplotlib.cm.ScalarMappable.set_norm
matplotlib.cm.ScalarMappable.set_clim
matplotlib.cbook.CallbackRegistry.connect
