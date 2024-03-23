"""
=============================================
Separate calculation and plotting of boxplots
=============================================

Drawing a `~.axes.Axes.boxplot` for a given data set, consists of two main operations,
that can also be used separately:

1. Calculating the boxplot statistics: `matplotlib.cbook.boxplot_stats`
2. Drawing the boxplot: `matplotlib.axes.Axes.bxp`

Thus, ``ax.boxplot(data)`` is equivalent to ::

    stats = cbook.boxplot_stats(data)
    ax.bxp(stats)

All styling keyword arguments are identical between `~.axes.Axes.boxplot` and
`~.axes.Axes.bxp`, and they are passed through from `~.axes.Axes.boxplot` to
`~.axes.Axes.bxp`. However, the *tick_labels* parameter of `~.axes.Axes.boxplot`
translates to a generic *labels* parameter in `.boxplot_stats`, because the labels are
data-related and attached to the returned per-dataset dictionaries.

The following code demonstrates the equivalence between the two methods.

"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cbook

np.random.seed(19680801)
data = np.random.randn(20, 3)

fig, (ax1, ax2) = plt.subplots(1, 2)

# single boxplot call
ax1.boxplot(data, tick_labels=['A', 'B', 'C'],
            patch_artist=True, boxprops={'facecolor': 'bisque'})

# separate calculation of statistics and plotting
stats = cbook.boxplot_stats(data, labels=['A', 'B', 'C'])
ax2.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'})

# %%
# Using the separate functions allows to pre-calculate statistics, in case you need
# them explicitly for other purposes, or to reuse the statistics for multiple plots.
#
# Conversely, you can also use the `~.axes.Axes.bxp` function directly, if you already
# have the statistical parameters:

fig, ax = plt.subplots()

stats = [
    dict(med=0, q1=-1, q3=1, whislo=-2, whishi=2, fliers=[-4, -3, 3, 4], label='A'),
    dict(med=0, q1=-2, q3=2, whislo=-3, whishi=3, fliers=[], label='B'),
    dict(med=0, q1=-3, q3=3, whislo=-4, whishi=4, fliers=[], label='C'),
]

ax.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'})

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bxp`
#    - `matplotlib.axes.Axes.boxplot`
#    - `matplotlib.cbook.boxplot_stats`
