"""
==============================
Add lines directly to a figure
==============================

You can add artists such as a `.Line2D` directly to a figure. This is
typically useful for visual structuring.

.. redirect-from:: /gallery/pyplots/fig_x
"""

import matplotlib.pyplot as plt

import matplotlib.lines as lines

fig, axs = plt.subplots(2, 2, gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
fig.add_artist(lines.Line2D([0, 1], [0.47, 0.47], linewidth=3))
fig.add_artist(lines.Line2D([0.5, 0.5], [1, 0], linewidth=3))
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.figure`
#    - `matplotlib.lines`
#    - `matplotlib.lines.Line2D`
