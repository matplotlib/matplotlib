"""
==================
Figure legend demo
==================

Instead of plotting a legend on each axis, a legend for all the artists on all
the sub-axes of a figure can be plotted instead.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2)

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')
fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='upper right')

plt.tight_layout()
plt.show()

# %%
# Sometimes we do not want the legend to overlap the Axes.  If you use
# *constrained layout*, you can specify "outside" at the beginning of
# the `loc` keyword argument. In this way, the legend is drawn outside
# the Axes on the (sub)figure.

# %%
# This example puts the legends at the "outside upper left" and
# "outside upper right".

fig, axs = plt.subplots(1, 2, layout='constrained')

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='outside upper left')
fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside upper right')

plt.show()

# %% Alternatively, we can put the legends at the "outside left
# upper" and "outside right upper". There is a slight difference
# in the exact location of the legends
# (see :ref:`legend_guide` for more details).

fig, axs = plt.subplots(1, 2, layout='constrained')

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='outside left upper')
fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside right upper')

plt.show()
