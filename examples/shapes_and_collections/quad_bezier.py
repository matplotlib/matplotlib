"""
============
Bezier Curve
============

This example showcases the `~.patches.PathPatch` object to create a Bezier
polycurve path patch.
"""

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig, ax = plt.subplots()
pp1 = mpatches.PathPatch(
    Path([(0, 0), (1, 0), (1, 1), (0, 0)],
         [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
    fc="none", transform=ax.transData)

ax.add_patch(pp1)
ax.plot([0.75], [0.25], "ro")
ax.set_title('The red point should be on the path')

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.path`
#    - `matplotlib.path.Path`
#    - `matplotlib.patches`
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.axes.Axes.add_patch`
