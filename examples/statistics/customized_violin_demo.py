"""
=================================
Demo of violin plot customization
=================================

This example demonstrates how to fully customize violin plots.
The first plot shows the default style by providing only
the data. The second plot first limits what matplotlib draws
with additional kwargs. Then a simplified representation of
a box plot is drawn on top. Lastly, the styles of the artists
of the violins are modified.

For more information on violin plots, the scikit-learn docs have a great
section: http://scikit-learn.org/stable/modules/density.html
"""

import matplotlib.pyplot as plt
import numpy as np


def adjacent_values(vals):
    q1, q3 = np.percentile(vals, [25, 75])
    # inter-quartile range iqr
    iqr = q3 - q1
    # upper adjacent values
    uav = q3 + iqr * 1.5
    uav = np.clip(uav, q3, vals[-1])
    # lower adjacent values
    lav = q1 - iqr * 1.5
    lav = np.clip(lav, vals[0], q1)
    return [lav, uav]


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# create test data
np.random.seed(123)
dat = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

# plot the default violin
ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(dat)

# customized violin
ax2.set_title('Customized violin plot')
parts = ax2.violinplot(
        dat, showmeans=False, showmedians=False,
        showextrema=False)

# customize colors
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

# medians
med = [np.percentile(sarr, 50) for sarr in dat]
# inter-quartile ranges
iqr = [[np.percentile(sarr, 25), np.percentile(sarr, 75)] for sarr in dat]
# upper and lower adjacent values
avs = [adjacent_values(sarr) for sarr in dat]

# plot whiskers as thin lines, quartiles as fat lines,
# and medians as points
for i, median in enumerate(med):
    # whiskers
    ax2.plot([i + 1, i + 1], avs[i], '-', color='black', linewidth=1)
    # quartiles
    ax2.plot([i + 1, i + 1], iqr[i], '-', color='black', linewidth=5)
    # medians
    ax2.plot(
        i + 1, median, 'o', color='white',
        markersize=6, markeredgecolor='none')

# set style for the axes
labels = ['A', 'B', 'C', 'D']    # labels
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
