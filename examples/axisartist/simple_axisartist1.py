"""
========================================
Using axisartist to place an axis at y=0
========================================

Note that the following example can also be implemented without mpl_toolkits;
see :doc:`/gallery/ticks_and_spines/spine_placement_demo`.
"""

import matplotlib.pyplot as plt
from mpl_toolkits import axisartist

fig = plt.figure()
fig.subplots_adjust(right=0.85)
ax = axisartist.Subplot(fig, 1, 1, 1)
fig.add_subplot(ax)

# make some axis invisible
ax.axis["bottom", "top", "right"].set_visible(False)

# make an new axis along the first axis axis (x-axis) which pass
# through y=0.
ax.axis["y=0"] = ax.new_floating_axis(nth_coord=0, value=0,
                                      axis_direction="bottom")
ax.axis["y=0"].toggle(all=True)
ax.axis["y=0"].label.set_text("y = 0")

ax.set_ylim(-2, 4)

plt.show()
