"""
=======
Hatches
=======

Hatching (pattern filled polygons) is currently supported in the backends
PS, PDF, SVG and Agg. The backends OSX, WX and Cairo ignore hatching.

See also :doc:`/gallery/images_contours_and_fields/contourf_hatching` for
an example using `~.Axes.contourf`.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle

x = np.arange(1, 5)
y1 = np.arange(1, 5)
y2 = np.ones(y1.shape) * 4

fig = plt.figure()
axs = fig.subplot_mosaic([['bar1', 'patches'], ['bar2', 'patches']])

ax = axs['bar1']
ax.bar(x, y1, edgecolor='black', hatch="/")
ax.bar(x, y2, bottom=y1, edgecolor='black', hatch='//')

ax = axs['bar2']
ax.bar(x, y1, edgecolor='black', hatch=['-', '+', 'x', '\\'])
ax.bar(x, y2, bottom=y1, edgecolor='black', hatch=['*', 'o', 'O', '.'])

ax = axs['patches']
ax.add_patch(Ellipse((4, 50), 10, 10, fill=True, hatch='*', facecolor='y'))
ax.add_patch(Polygon([(10, 20), (30, 50), (50, 10)], hatch='\\/...',
                     facecolor='g'))
x = np.arange(0, 40, 0.2)
ax.fill_between(x, np.sin(x) * 4 + 30, y2=0, hatch='///', zorder=2, fc='c')
ax.set_xlim([0, 40])
ax.set_ylim([10, 60])
ax.set_aspect(1)
plt.show()

##############################################################################
# Hatch style reference
# ---------------------
#
# For convenience we document the 9 different possible hatch styles using the
# code below.  Note that repeating the hatch character increases the density of
# of the hatches, as shown in the first row and with "++".  Note also that
# hatches can be combined ("/+O" in the first row).

fig, axs = plt.subplots(3, 4, constrained_layout=True, figsize=(5, 4))

hatches = ['/', '///', '/+O', '\\',
           '|', '-', '++', 'x',
           'o', 'O', '.', '*']

for ax, h in zip(axs.flat, hatches):
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch=h))
    ax.text(1, -0.5, f"' {h} '", size=10, ha="center")
    ax.axis('equal')
    ax.axis('off')
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
matplotlib.patches
matplotlib.patches.Ellipse
matplotlib.patches.Polygon
matplotlib.axes.Axes.add_patch
matplotlib.patches.Patch.set_hatch
matplotlib.axes.Axes.bar
matplotlib.pyplot.bar
