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
adelie_means = (188.80, 189.73, 191.20)
gentoo_means = (217.19, 0, 0)
chinstrap_means = (0, 48.83, 0)

x = np.arange(len(islands))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, adelie_means, width, label='Adelie')
rects2 = ax.bar(x, gentoo_means, width, label='Chinstrap')
rects3 = ax.bar(x + width, chinstrap_means, width, label='Gentoo')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Flipper length (mm)')
ax.set_title('Average flipper length of penguin species by island')
ax.set_xticks(x, islands)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

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
