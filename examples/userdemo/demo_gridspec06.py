r"""
================
Nested GridSpecs
================

This example demonstrates the use of nested `.GridSpec`\s.
"""

import matplotlib.pyplot as plt
import numpy as np


def squiggle_xy(a, b, c, d):
    i = np.arange(0.0, 2*np.pi, 0.05)
    return np.sin(i*a)*np.cos(i*b), np.sin(i*c)*np.cos(i*d)


fig = plt.figure(figsize=(8, 8))
outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)

for a in range(4):
    for b in range(4):
        # gridspec inside gridspec
        inner_grid = outer_grid[a, b].subgridspec(3, 3, wspace=0, hspace=0)
        for c in range(3):
            for d in range(3):
                ax = fig.add_subplot(inner_grid[c, d])
                ax.plot(*squiggle_xy(a + 1, b + 1, c + 1, d + 1))
                ax.set(xticks=[], yticks=[])

# show only the outside spines
for ax in fig.get_axes():
    ax.spines['top'].set_visible(ax.is_first_row())
    ax.spines['bottom'].set_visible(ax.is_last_row())
    ax.spines['left'].set_visible(ax.is_first_col())
    ax.spines['right'].set_visible(ax.is_last_col())

plt.show()
