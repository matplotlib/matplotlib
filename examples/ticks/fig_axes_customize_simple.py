"""
=========================
Fig Axes Customize Simple
=========================

Customize the background, labels and ticks of a simple plot.

.. redirect-from:: /gallery/pyplots/fig_axes_customize_simple
"""

import matplotlib.pyplot as plt

# %%
# `.pyplot.figure` creates a `matplotlib.figure.Figure` instance.

fig = plt.figure()
rect = fig.patch  # a rectangle instance
rect.set_facecolor('lightgoldenrodyellow')

ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
rect = ax1.patch
rect.set_facecolor('lightslategray')

ax1.tick_params(axis='x', labelcolor='tab:red', labelrotation=45, labelsize=16)
ax1.tick_params(axis='y', color='tab:green', size=25, width=3)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axis.Axis.get_ticklabels`
#    - `matplotlib.axis.Axis.get_ticklines`
#    - `matplotlib.text.Text.set_rotation`
#    - `matplotlib.text.Text.set_fontsize`
#    - `matplotlib.text.Text.set_color`
#    - `matplotlib.lines.Line2D`
#    - `matplotlib.lines.Line2D.set_markeredgecolor`
#    - `matplotlib.lines.Line2D.set_markersize`
#    - `matplotlib.lines.Line2D.set_markeredgewidth`
#    - `matplotlib.patches.Patch.set_facecolor`
