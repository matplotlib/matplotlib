"""
============
Ellipse Demo
============

Draw many ellipses. Here individual ellipses are drawn. Compare this
to the :doc:`Ellipse collection example
</gallery/shapes_and_collections/ellipse_collection>`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


# Fixing random state for reproducibility
np.random.seed(19680801)

NUM = 250

ells = [Ellipse(xy=np.random.rand(2) * 10,
                width=np.random.rand(), height=np.random.rand(),
                angle=np.random.rand() * 360)
        for i in range(NUM)]

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(np.random.rand())
    e.set_facecolor(np.random.rand(3))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.show()

#############################################################################
# ===============
# Ellipse Rotated
# ===============
#
# Draw many ellipses with different angles.
#

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

angle_step = 45  # degrees
angles = np.arange(0, 360, angle_step)

ax = plt.subplot(aspect='equal')

for angle in angles:
    ellipse = Ellipse((0, 0), 4, 2, angle=angle, alpha=0.1)
    ax.add_artist(ellipse)

plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)

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
matplotlib.axes.Axes.add_artist
matplotlib.artist.Artist.set_clip_box
matplotlib.artist.Artist.set_alpha
matplotlib.patches.Patch.set_facecolor
