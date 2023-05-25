"""
.. _patches_artists:

=======
Patches
=======

:mod:`Patches <matplotlib.patches>` are a family of Artists that can be used
when drawing arbitrary two-dimensional regions. In addition to the general
Patch Artists :class:`~.patches.PathPatch` and :class:`~.patches.Polygon`,
common shapes have corresponding Patch Artists such as
:class:`~.patches.Circle`, :class:`~.patches.Rectangle`,
:class:`~.patches.Ellipse`, etc.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fig, ax = plt.subplots()

polygon = mpl.patches.Polygon(np.array([[1, 0], [0.5, 1.5], [2, 1]]), closed=True)
ax.add_patch(polygon)

circle = mpl.patches.Circle((2, 2), 0.5, facecolor='orange', edgecolor='black')
ax.add_patch(circle)

ax.set(xlim=(0, 3), ylim=(0, 3), box_aspect=1)
