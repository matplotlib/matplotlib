"""
=========================
Fig Axes Customize Simple
=========================

Customize the background, labels and ticks of a simple plot.

.. redirect-from:: /gallery/pyplots/fig_axes_customize_simple
"""

import matplotlib.pyplot as plt

fig = plt.figure()
fig.patch.set_facecolor('lightgoldenrodyellow')

ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
ax1.patch.set_facecolor('lightslategray')

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
#    - `matplotlib.axes.Axes.tick_params`
#    - `matplotlib.patches.Patch.set_facecolor`
