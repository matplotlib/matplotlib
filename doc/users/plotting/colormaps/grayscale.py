'''
Show what matplotlib colormaps look like in grayscale.
Uses lightness L* as a proxy for grayscale value.
'''

from colormaps import cmaps

#from skimage import color
# we are using a local copy of colorconv from scikit-image to reduce dependencies.
# You should probably use the one from scikit-image in most cases.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from colorspacious import cspace_converter

mpl.rcParams.update({'font.size': 14})


# indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

def plot_color_gradients(cmap_category, cmap_list):
    nrows = len(cmap_list)
    fig, axes = plt.subplots(nrows=nrows, ncols=2)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99,
                        wspace=0.05)
    fig.suptitle(cmap_category + ' colormaps', fontsize=14, y=1.0, x=0.6)

    for ax, name in zip(axes, cmap_list):

        # Get rgb values for colormap
        rgb = cm.get_cmap(plt.get_cmap(name))(x)[np.newaxis,:,:3]

        # Get colormap in CAM02-UCS colorspace. We want the lightness.
        lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
        L = lab[0,:,0]
        L = np.float32(np.vstack((L, L, L)))

        ax[0].imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)
        pos = list(ax[0].get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax[0].set_axis_off()
        ax[1].set_axis_off()
    plt.show()


for cmap_category, cmap_list in cmaps:

    plot_color_gradients(cmap_category, cmap_list)
