"""
=======================
Align labels and titles
=======================

Aligning xlabel, ylabel, and title using `.Figure.align_xlabels`,
`.Figure.align_ylabels`, and `.Figure.align_titles`.

`.Figure.align_labels` wraps the x and y label functions.

We align the xlabels and ylabels using short calls to `.Figure.align_xlabels`
and `.Figure.align_ylabels`. We also show a manual way to align the ylabels
using the `~.Axis.set_label_coords` method of the YAxis object. Note this requires
knowing a good offset value which is hardcoded.

Note that the labels "XLabel1 0" and "XLabel1 1" would normally be much closer
to their respective x-axes, "YLabel1 1" and "YLabel1 2" would be much closer to their
respective y-axes, and titles "YLabels Aligned" and "YLabels Manually Aligned" would
be much closer to their corresponding subplots.

.. redirect-from:: /gallery/pyplots/align_ylabels
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
fig.subplots_adjust(left=0.2, wspace=0.6)
box = dict(facecolor='yellow', pad=5, alpha=0.2)

# First column: Misaligned labels
axs[0, 0].plot(2000 * np.random.rand(10))
axs[0, 0].set_title('YLabels Not Aligned')
axs[0, 0].xaxis.tick_top()
axs[0, 0].set_ylabel('YLabel0 0', bbox=box)
axs[0, 0].set_ylim(0, 2000)

axs[1, 0].plot(np.random.rand(10))
axs[1, 0].set_xlabel('XLabel1 0', bbox=box)
axs[1, 0].set_ylabel('YLabel1 0', bbox=box)

# Second column: Automatically aligned labels
axs[0, 1].plot(2000 * np.random.rand(10))
axs[0, 1].set_title('YLabels Aligned')
axs[0, 1].set_ylabel('YLabel0 1', bbox=box)
axs[0, 1].set_ylim(0, 2000)

axs[1, 1].plot(np.random.rand(10))
axs[1, 1].set_xlabel('XLabel1 1', bbox=box)
axs[1, 1].set_ylabel('YLabel1 1', bbox=box)

# Third column: Manually adjusted labels
axs[0, 2].plot(2000 * np.random.rand(10))
axs[0, 2].set_title('YLabels Manually Aligned')
axs[0, 2].set_ylabel('YLabel0 2', bbox=box)
axs[0, 2].set_ylim(0, 2000)

axs[1, 2].plot(np.random.rand(10))
axs[1, 2].tick_params(axis='x', rotation=55)
axs[1, 2].set_xlabel('XLabel1 2', bbox=box)
axs[1, 2].set_ylabel('YLabel1 2', bbox=box)

# Align labels
fig.align_ylabels(axs[:, 1])  # Align only the second column's y-labels
fig.align_xlabels()           # Align all x-axis labels
fig.align_titles()            # Align titles

# Manually adjust y-labels for the third column
for ax in axs[:, 2]:
    ax.yaxis.set_label_coords(-0.3, 0.5)

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.align_xlabels`
#    - `matplotlib.figure.Figure.align_ylabels`
#    - `matplotlib.figure.Figure.align_labels`
#    - `matplotlib.figure.Figure.align_titles`
#    - `matplotlib.axis.Axis.set_label_coords`
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_ylabel`
#    - `matplotlib.axes.Axes.set_ylim`

# %%
# .. tags::
#
#    component: label
#    component: title
#    styling: position
#    level: beginner
