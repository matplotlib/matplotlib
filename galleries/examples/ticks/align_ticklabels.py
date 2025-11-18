"""
=================
Align tick labels
=================

By default, tick labels are aligned towards the axis. This means the set of
*y* tick labels appear right-aligned. Because the alignment reference point
is on the axis, left-aligned tick labels would overlap the plotting area.
To achieve a good-looking left-alignment, you have to additionally increase
the padding.
"""
import matplotlib.pyplot as plt

population = {
    "Sydney": 5.2,
    "Mexico City": 8.8,
    "São Paulo": 12.2,
    "Istanbul": 15.9,
    "Lagos": 15.9,
    "Shanghai": 21.9,
}

fig, ax = plt.subplots(layout="constrained")
ax.barh(population.keys(), population.values())
ax.set_xlabel('Population (in millions)')

# left-align all ticklabels
for ticklabel in ax.get_yticklabels():
    ticklabel.set_horizontalalignment("left")

# increase padding
ax.tick_params("y", pad=70)
