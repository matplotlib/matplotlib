"""
=================
Stacked bar chart
=================

This is an example of creating a stacked bar plot
using `~matplotlib.pyplot.bar`.
"""

import matplotlib.pyplot as plt

# data from https://allisonhorst.github.io/palmerpenguins/

species = ("Adelie", "Gentoo", "Chinstrap")
booleans = ["True", "False"]
above_average_weight = (70, 31, 58)
below_average_weight = (82, 37, 66)
weight_counts = [above_average_weight, below_average_weight]
width = 0.5

fig, ax = plt.subplots()

for boolean, weight_count in zip(booleans, weight_counts):
    if boolean == "True":
        p = ax.bar(species, weight_count, width, label=boolean)

    else:
        p = ax.bar(species, weight_count, width, label=boolean,
                    bottom=weight_counts[0])

ax.set_title("Number of penguins with above average body mass")
ax.legend(loc="upper right")

plt.show()
