"""
*************************************************
Manipulating and Creating Colormaps in Matplotlib
*************************************************

Matplotlib colormaps are implimented as a class, which makes them quite
flexible, but ocasionally opaque to users as to how to create and/or
manipulate them.  This opacity is not helped in the library by the fact that
the named colormaps are accessed via `.matplotlib.cm.get_cmap` module, whereas
the colormap class itself is defined in `.matplotlin.colors.Colormap`!

Getting colormaps and accessing their values
============================================

First, getting a named colormap, most of which are listed in
:doc:`/tutorials/colors/colormaps` requires the use of
`.matplotlib.cm.get_cmap`, which returns a
:class:`.matplotlib.colors.ListedColormap` object.  The second argument gives
the size of the list of colors used to define the colormap.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict

viridis = cm.get_cmap('viridis', 12)
print(viridis)

##############################################################################
# This list of colors can be directly accessed using the ``colors`` property,
# or it can be indirectly acccessed by calling the object.  Note that the list
# is of the form of an RGBA Nx4 array, where N is the length of the colormap.

print('viridis.colors', viridis.colors)
print('viridis(range(12))', viridis(range(12)))
print('viridis(np.linspace(0, 1, 12))', viridis(np.linspace(0, 1, 12)))

##############################################################################
# The colormap is a lookup table, so "oversampling" the colormap returns
# nearest-neighbor interpolation (note the repeated colors in the list below)

print('viridis(np.linspace(0, 1, 15))', viridis(np.linspace(0, 1, 15)))

##############################################################################
# Creating a new ListedColormap: Colormap carpentry
# =================================================
#
# This is essential the inverse operation of the above where we supply a
# Nx4 numpy array with all values between 0 and 1,
# to `.ListedColormap` to make a new colormap.  This means that
# any numpy operations that we can do on a Nx4 array make carpentry of
# new colormaps from existing colormaps quite straight forward.
#
# Suppose we want to make the first 25 entries of a 256-length "viridis"
# colormap pink for some reason:

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)

def plot_examples(cms):
    np.random.seed(19680801)
    data = np.random.randn(30,30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cm] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cm, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

plot_examples([viridis, newcmp])

##############################################################################
# We can easily reduce the range of a colormap; here we choose the middle
# 0.5 of the colormap.

viridis = cm.get_cmap('viridis', 256)
newcmp = ListedColormap(viridis(np.linspace(0.25, 0.75, 256)))
plot_examples([viridis, newcmp])

##############################################################################
# and we can easily paste together two colormaps:

top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')
plot_examples([viridis, newcmp])

##############################################################################
# LinearSegmented colormaps
# =========================
#
# LinearSegmented colormaps are an alternate way to specify colormaps that
# specify anchor points for linear ramps for each of RGB and optionally, alpha
# (RGBA).
#
# The format to specify these colormaps is a bit complicated to allow
# discontinuities at the anchor points.  First, with no discontinuities:

cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.75, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  1.0, 1.0]]}

def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()

plot_linearmap(cdict)

#############################################################################
# However, consider the case where the third column is different than the
# second.  The linear interpolation between red[i, 0] and red[i+1, 0] is
# from red[i, 2] to red[i+1, 1].  This format allows us to have
# discontinuities in the colormap at the anchor points; in this case
# between 0 and 0.5, the linear interpolation goes from 0.3 to 1, and
# between 0.5 and 1 it goes from 0.9 to 1.  Note that red[0, 1], and red[2, 2]
# are both superfluous to the interpolation, which happens between the last
# element of the first anchor and the first element of the second anchor.

cdict['red'] =   [[0.0,  0.0, 0.3],
                   [0.5,  1.0, 0.9],
                   [1.0,  1.0, 1.0]]
plot_linearmap(cdict)


#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.axes.Axes.pcolormesh
matplotlib.figure.Figure.colorbar
matplotlib.colors
matplotlib.colors.LinearSegmentedColormap
matplotlib.colors.ListedColormap
matplotlib.cm
matplotlib.cm.ScalarMappable.get_cmap
matplotlib.pyplot.register_cmap
matplotlib.cm.register_cmap
