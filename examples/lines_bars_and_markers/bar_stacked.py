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
species = ["Adelie", "Gentoo", "Chinstrap"]
adelie_means = (44, 56, 52)
gentoo_means = (124, 0, 0)
chinstrap_means = (0, 68, 0)
penguin_means = [adelie_means, gentoo_means, chinstrap_means]
width = 0.5

fig, ax = plt.subplots()

for species, penguin_mean in zip(species, penguin_means):
    if species == "Adelie":
        p = ax.bar(islands, penguin_mean, width, label=species)

    else:
        p = ax.bar(islands, penguin_mean, width, label=species,
                    bottom=penguin_means[0])

ax.set_title("Number of penguins by island")
ax.legend(loc="upper right")

plt.show()
