"""
================
Axis line styles
================

This example shows some configurations for axis style.

Note: The `mpl_toolkits.axisartist` axes classes may be confusing for new
users. If the only aim is to obtain arrow heads at the ends of the axes,
rather check out the :doc:`/gallery/recipes/centered_spines_with_arrows`
example.
"""

from mpl_toolkits.axisartist.axislines import Subplot
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = Subplot(fig, 111)
fig.add_subplot(ax)

for direction in ["bottom", "left"]:  # Add arrows for bottom and left axes.
    ax.axis[direction].set_axisline_style("-|>")

for direction in ["top", "right"]:  # Hide top and right axes.
    ax.axis[direction].set_visible(False)

x = np.linspace(-0.5, 1., 100)
ax.plot(x, np.sin(x*np.pi))

plt.show()
