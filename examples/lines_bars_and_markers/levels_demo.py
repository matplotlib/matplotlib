"""
=============
Histline Demo
=============

This example demonstrates the use of `~.matplotlib.pyplot.levels`
for histogram and histogram-like data visualization and an associated
underlying `.LevelsPatch` artist, which is
a contrained version of `.PathPatch` specified by its bins and edges.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import LevelsPatch

np.random.seed(0)
h, bins = np.histogram(np.random.normal(5, 3, 5000),
                       bins=np.linspace(0, 10, 20))

fig, axs = plt.subplots(3, 1, figsize=(7, 15))
axs[0].levels(h, bins, label='Simple histogram')
axs[0].levels(h, bins+5, baseline=50, label='--//-- w/ modified baseline')
axs[0].levels(h, bins+10, baseline=None, label='--//-- w/ no edges')
axs[0].set_title("Step Histograms")

axs[1].levels(np.arange(1, 6, 1), fill=True,
              label='Filled histogram\nw/ automatatic edges')
axs[1].levels(np.arange(1, 6, 1)*0.3, np.arange(2, 8, 1),
              orientation='horizontal', hatch='//',
              label='Hatched histogram\nw/ horizontal orientation')
axs[1].set_title("Filled histogram")

patch = LevelsPatch(values=[1, 2, 3, 2, 1],
                    edges=range(1, 7),
                    label=('Patch derived underlying object\n'
                           'with default edge/facecolor behaviour'))
axs[2].add_patch(patch)
axs[2].set_xlim(0, 7)
axs[2].set_ylim(-1, 5)
axs[2].set_title("LevelsPatch artist")

for ax in axs:
    ax.legend()
plt.show()


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
matplotlib.axes.Axes.levels
matplotlib.pyplot.levels
matplotlib.patches.LevelsPatch
