"""
===========================================
Selecting individual colors from a colormap
===========================================

Sometimes we want to use more colors or a different set of colors than the default color
cycle provides. Selecting individual colors from one of the provided colormaps can be a
convenient way to do this.

Once we have hold of a `.Colormap` instance, the individual colors can be accessed
by passing it an index.  If we want a specific number of colors taken at regular
intervals from a continuous colormap, we can create a new colormap using the
`~.Colormap.resampled` method.

For more details about manipulating colormaps, see :ref:`colormap-manipulation`.
"""

import matplotlib.pyplot as plt

import matplotlib as mpl

n_lines = 21

cmap = mpl.colormaps.get_cmap('plasma').resampled(n_lines)

fig, ax = plt.subplots(layout='constrained')

for i in range(n_lines):
    ax.plot([0, i], color=cmap(i))

plt.show()

# %%
# Instead of passing colors one by one to `~.Axes.plot`, we can replace the default
# color cycle with a different set of colors.  Specifying a `~cycler.cycler` instance
# within `.rcParams` achieves that.  See :ref:`color_cycle` for details.


from cycler import cycler

cmap = mpl.colormaps.get_cmap('Dark2')
colors = cmap(range(cmap.N))  # cmap.N is number of unique colors in the colormap

with mpl.rc_context({'axes.prop_cycle': cycler(color=colors)}):

    fig, ax = plt.subplots(layout='constrained')

    for i in range(n_lines):
        ax.plot([0, i])

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.resampled`
