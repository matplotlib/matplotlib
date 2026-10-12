"""
============
Pareto chart
============

A Pareto chart combines a bar chart of counts, sorted in descending order,
with a line showing the cumulative percentage. It is used to identify the
few categories that account for most of the total (the "vital few").

The cumulative line is drawn on a secondary y-axis created with
`.Axes.twinx`, and a dashed reference line at 80% marks the usual Pareto
threshold.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import PercentFormatter

# Mock data: number of defects found per category.
categories = ["Scratches", "Dents", "Misalignment", "Wrong color",
              "Missing parts", "Cracks", "Other"]
counts = np.array([58, 42, 17, 9, 6, 4, 3])

# Sort in descending order and compute the cumulative percentage.
order = np.argsort(counts)[::-1]
categories = [categories[i] for i in order]
counts = counts[order]
cumulative_pct = np.cumsum(counts) / counts.sum() * 100

x = np.arange(len(categories))

fig, ax = plt.subplots(layout='constrained')

ax.bar(x, counts, color='tab:blue')
ax.set_xticks(x, categories, rotation=30, ha='right')
ax.set_ylabel('Number of defects')
ax.set_title('Pareto chart of defects')

ax2 = ax.twinx()  # second Axes sharing the same x-axis
ax2.plot(x, cumulative_pct, color='tab:red', marker='o')
ax2.axhline(80, color='tab:gray', linestyle='--', linewidth=1)
ax2.set_ylim(0, 105)  # headroom so the 100% marker is not clipped
ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax2.set_ylabel('Cumulative percentage')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.twinx` / `matplotlib.pyplot.twinx`
#    - `matplotlib.axes.Axes.set_xticks` / `matplotlib.pyplot.xticks`
#    - `matplotlib.ticker.PercentFormatter`
#
# .. tags::
#
#    component: axes
#    plot-type: bar
#    plot-type: line
#    level: beginner
