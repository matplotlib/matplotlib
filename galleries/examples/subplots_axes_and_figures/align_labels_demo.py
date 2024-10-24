"""
=======================
Align labels and titles
=======================

Aligning xlabel, ylabel, and title using `.Figure.align_xlabels`,
`.Figure.align_ylabels`, and `.Figure.align_titles`.

`.Figure.align_labels` wraps the x and y label functions.

Note that the xlabel "XLabel1 1" would normally be much closer to the
x-axis, "YLabel0 0" would be much closer to the y-axis, and title
"Title0 0" would be much closer to the top of their respective axes.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, layout='constrained')

for i in range(2):
    for j in range(2):
        ax = axs[i][j]
        ax.plot(np.arange(1., 0., -0.1) * 1000., np.arange(1., 0., -0.1))
        ax.set_title(f'Title {i} {j}')
        ax.set_xlabel(f'XLabel {i} {j}')
        ax.set_ylabel(f'YLabel {i} {j}')
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            if i == 0 and j == 1:
                ax.xaxis.tick_top()
            ax.set_xticks(np.linspace(0, 1000, 5))
            ax.set_xticklabels([250 * n for n in range(5)])
            ax.xaxis.set_tick_params(rotation=55, rotation_mode='xtick')

fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
fig.align_titles()

plt.show()

# %%
# .. tags::
#
#    component: label
#    component: title
#    styling: position
#    level: beginner
