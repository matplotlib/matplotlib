"""
==============
Legend outside
==============

.. redirect-from:: /gallery/userdemo/simple_legend01

To move a legend outside the Axes, one can position it manually via the anchor
point ``bbox_to_anchor`` (in Axes coordinates), and then use ``loc`` to determine
where the anchor point is relative to the legend.

In the example, we specify that the upper left edge of the legend (
``loc='upper left'``) is at ``bbox_to_anchor=(1.05, 1)``, i.e. vertically at the
top of the Axes and just a bit outside to the right.

We additionally set ``borderaxespad=0``, so that there is no extra space around the
legend box.

By default, Axes fill the whole figure area, and thus legends placed outside would
be cut of. We therefore use a layout manager to resize everything so that the legend
fits into the figure. Either of the 'constrained' or 'tight' layout managers will
work.
"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(layout='constrained')
ax.plot([1, 2, 3], label="line 1")
ax.plot([3, 2, 1], label="line 2")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()

###############################################################################
# Similarly, one can place the legend at the top or bottom. In this case, it's
# reasonable to place the entries horizontally, by using as many columns as
# legend entries (here ``ncol=2``). Additionally, ``mode="expand"`` makes sure
# the legend spans the full width of the Axes.

fig, ax = plt.subplots(layout='constrained')
ax.plot([1, 2, 3], label="line 1")
ax.plot([3, 2, 1], label="line 2")
ax.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc='lower left',
          ncols=2, mode="expand", borderaxespad=0)
plt.show()
