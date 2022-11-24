"""
=============================
Grouped bar chart with labels
=============================

This example shows a how to create a grouped bar chart and how to annotate
bars with labels.
"""

# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

species = ("Adelie", "Chinstrap", "Gentoo")
measurements = ['Bill Length', 'Bill Depth']
bill_length = (38.80, 48.83, 47.50)
bill_depth = (18.34, 18.43, 14.98)
penguin_means = [bill_length, bill_depth]

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
fig, ax = plt.subplots()

for measurement, penguin_mean in zip(measurements, penguin_means):
    offset = (width * multiplier)
    rects = ax.bar(x + offset, penguin_mean, width, label=measurement)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + (width / 2), species)
ax.legend()

fig.tight_layout()

plt.ylim(0, 60)
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
