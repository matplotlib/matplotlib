"""
================
Patch hatchcolor
================

This example shows how to use the *hatchcolor* parameter to set the color of
the hatch. The *hatchcolor* parameter is available for `~.patches.Patch`,
child classes of Patch, and methods that pass through to Patch.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

fig, (ax1, ax2) = plt.subplots(1, 2)

# Rectangle with red hatch color and black edge color
ax1.add_patch(Rectangle((0.1, 0.5), 0.8, 0.3, hatch=".", hatchcolor='red',
                        edgecolor='black', lw=2))
# If hatchcolor is not passed, the hatch will match the edge color
ax1.add_patch(Rectangle((0.1, 0.1), 0.8, 0.3, hatch='x', edgecolor='orange', lw=2))

x = np.arange(1, 5)
y = np.arange(1, 5)

ax2.bar(x, y, facecolor='none', edgecolor='red', hatch='//', hatchcolor='blue')
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 5)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Polygon`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
