"""
============
Pareto chart
============

A Pareto chart combines a bar chart of counts, sorted in descending order,
with a line showing the cumulative percentage. It is used to identify the
few categories that account for most of the total (the "vital few").

The cumulative line is drawn against the same y-axis as the bars, so the left
axis runs from 0 to the total count. A secondary y-axis created with
`.Axes.secondary_yaxis` relabels the same scale as a percentage, keeping both
axes aligned, and a dashed reference line at 80% marks the usual Pareto
threshold.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import PercentFormatter

# Mock data: number of defects found per category.
categories = ["Scratches", "Dents", "Misalignment", "Wrong color",
              "Missing parts", "Cracks", "Other"]
counts = np.array([58, 42, 17, 9, 6, 4, 3])

# Sort in descending order.
order = np.argsort(counts)[::-1]
categories = [categories[i] for i in order]
counts = counts[order]
total = counts.sum()

x = np.arange(len(categories))

fig, ax = plt.subplots(layout='constrained')

ax.bar(x, counts, color='tab:blue')
ax.set_xticks(x, categories, rotation=30, ha='right')
ax.set_ylabel('Number of defects')
ax.set_title('Pareto chart of defects')

ax.plot(x, np.cumsum(counts), color='tab:red', marker='o')
ax.axhline(0.8 * total, color='tab:gray', linestyle='--', linewidth=1)
ax.set_ylim(0, total * 1.05)  # headroom so the 100% marker is not clipped

# Secondary axis that maps counts to a percentage of the total (and back).
secax = ax.secondary_yaxis('right', functions=(lambda c: c / total * 100,
                                                lambda p: p / 100 * total))
secax.yaxis.set_major_formatter(PercentFormatter())
secax.set_ylabel('Cumulative percentage')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.secondary_yaxis`
#    - `matplotlib.axes.Axes.set_xticks` / `matplotlib.pyplot.xticks`
#    - `matplotlib.ticker.PercentFormatter`
#
# .. tags::
#
#    component: axes
#    plot-type: bar
#    plot-type: line
#    level: beginner
