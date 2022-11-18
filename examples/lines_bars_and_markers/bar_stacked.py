"""
=================
Stacked bar chart
=================

This is an example of creating a stacked bar plot
using `~matplotlib.pyplot.bar`.
"""

import matplotlib.pyplot as plt

# data from https://allisonhorst.github.io/palmerpenguins/

islands = ("Biscoe", "Dream", "Torgersen")
adelie_means = (44, 56, 52)
gentoo_means = (124, 0, 0)
chinstrap_means = (0, 68, 0)
width = 0.5

fig, ax = plt.subplots()

p1 = ax.bar(islands, adelie_means, width, label="Adelie")
p2 = ax.bar(islands, gentoo_means, width, bottom=adelie_means, label="Gentoo")
p3 = ax.bar(islands, chinstrap_means, width,
              bottom=adelie_means, label="Chinstrap")

ax.set_title("Number of penguins by island")
ax.legend(loc="upper right")

plt.show()
