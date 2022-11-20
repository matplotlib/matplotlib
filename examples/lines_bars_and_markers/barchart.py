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

islands = ('Biscoe', 'Dream', 'Torgersen')
species = ["Adelie", "Gentoo", "Chinstrap"]
adelie_means = (188.80, 189.73, 191.20)
gentoo_means = (217.19, 0, 0)
chinstrap_means = (0, 48.83, 0)
penguin_means = [adelie_means, gentoo_means, chinstrap_means]

x = np.arange(len(islands))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
fig, ax = plt.subplots()

for specie, penguin_mean in zip(species, penguin_means):
    rects = ax.bar(x + (width * multiplier), penguin_mean, width, label=specie)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Flipper length (mm)')
ax.set_title('Average flipper length of penguin species by island')
ax.set_xticks(x, islands)
ax.legend()

fig.tight_layout()

plt.ylim(0, 250)
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
