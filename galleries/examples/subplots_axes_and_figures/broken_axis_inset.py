"""
=================
Broken axis inset
=================

Example of a broken axis with an inset spanning across the break.
"""

import matplotlib.pyplot as plt

# initialize figure
fig, axes = plt.subplots(ncols=2, sharey=True, gridspec_kw={'top': 0.6})
inset = fig.add_axes([0.2, 0.7, 0.6, 0.2])

# set axes limits
axes[0].set_xlim(0, 0.5)
axes[1].set_xlim(0.5, 1)
axes[1].set_ylim(0, 1)
inset.set_xlim(0.3, 0.7)
inset.set_ylim(0.4, 0.6)

# hide the spines between the axes
axes[0].spines.right.set_visible(False)
axes[1].spines.left.set_visible(False)
axes[1].yaxis.tick_right()

# indicate inset on the left axes
indicator = axes[0].indicate_inset(inset_ax=inset, edgecolor='tab:red')
indicator.rectangle.set_clip_box(axes[0].bbox)
indicator.rectangle.set_clip_on(True)
indicator.connectors[2].set_visible(False)

# indicate inset on the right axes
indicator = axes[1].indicate_inset(inset_ax=inset, edgecolor='tab:blue')
indicator.rectangle.set_clip_box(axes[1].bbox)
indicator.rectangle.set_clip_on(True)
indicator.connectors[0].set_visible(False)

# show
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.indicate_inset_zoom`
#    - `matplotlib.inset.InsetIndicator`
#
# .. tags::
#
#    component: axes
#    level: advances
