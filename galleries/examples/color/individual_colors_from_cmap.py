"""
===========================================
Selecting individual colors from a colormap
===========================================

Sometimes we want to use more colors or a different set of colors than the default color
cycle provides. Selecting individual colors from one of the provided colormaps can be a
convenient way to do this.

We can retrieve colors from any `.Colormap` by calling it with a float or a list of
floats in the range [0, 1]; e.g. ``cmap(0.5)`` will give the middle color. See also
`.Colormap.__call__`.

Extracting colors from a continuous colormap
--------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

n_lines = 21
cmap = mpl.colormaps['plasma']

# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, n_lines))

fig, ax = plt.subplots(layout='constrained')

for i, color in enumerate(colors):
    ax.plot([0, i], color=color)

plt.show()

# %%
#
# Extracting colors from a discrete colormap
# ------------------------------------------
# The list of all colors in a `.ListedColormap` is available as the ``colors``
# attribute.

colors = mpl.colormaps['Dark2'].colors

fig, ax = plt.subplots(layout='constrained')

for i, color in enumerate(colors):
    ax.plot([0, i], color=color)

plt.show()

# %%
# See Also
# --------
#
# For more details about manipulating colormaps, see :ref:`colormap-manipulation`.  To
# change the default color cycle, see :ref:`color_cycle`.
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.resampled`
