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


# functions to calculate percentiles and adjacent values
def percentile(vals, p):
    N = len(vals)
    n = p*(N+1)
    k = int(n)
    d = n-k
    if k <= 0:
        return vals[0]
    if k >= N:
        return vals[N-1]
    return vals[k-1] + d*(vals[k] - vals[k-1])


def adjacent_values(vals):
    q1 = percentile(vals, 0.25)
    q3 = percentile(vals, 0.75)
    iqr = q3 - q1  # inter-quartile range

    # upper adjacent values
    uav = q3 + iqr * 1.5
    if uav > vals[-1]:
        uav = vals[-1]
    if uav < q3:
        uav = q3

    # lower adjacent values
    lav = q1 - iqr * 1.5
    if lav < vals[0]:
        lav = vals[0]
    if lav > q1:
        lav = q1
    return [lav, uav]


# create test data
np.random.seed(123)
dat = [np.random.normal(0, std, 100) for std in range(1, 5)]
lab = ['A', 'B', 'C', 'D']    # labels
med = []    # medians
iqr = []    # inter-quantile ranges
avs = []    # upper and lower adjacent values
for arr in dat:
    sarr = sorted(arr)
    med.append(percentile(sarr, 0.5))
    iqr.append([percentile(sarr, 0.25), percentile(sarr, 0.75)])
    avs.append(adjacent_values(sarr))

# plot the violins
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4),
                               sharey=True)
_ = ax1.violinplot(dat)
parts = ax2.violinplot(dat, showmeans=False, showmedians=False,
                       showextrema=False)

ax1.set_title('Default violin plot')
ax2.set_title('Customized violin plot')

# plot whiskers as thin lines, quartiles as fat lines,
# and medians as points
for i in range(len(med)):
    # whiskers
    ax2.plot([i + 1, i + 1], avs[i], '-', color='black', linewidth=1)
    ax2.plot([i + 1, i + 1], iqr[i], '-', color='black', linewidth=5)
    ax2.plot(i + 1, med[i], 'o', color='white',
             markersize=6, markeredgecolor='none')

# customize colors
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

ax1.set_ylabel('Observed values')
for ax in [ax1, ax2]:
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(lab) + 1))
    ax.set_xticklabels(lab)
    ax.set_xlim(0.25, len(lab) + 0.75)
    ax.set_xlabel('Sample name')

plt.subplots_adjust(bottom=0.15, wspace=0.05)

plt.show()
