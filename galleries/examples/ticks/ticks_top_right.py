"""
=================================
Ticks and labels on top and right
=================================

Demonstrate moving ticks, tick labels, and axis labels
to the top or right side of an Axes.

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(10))

# Move ticks and tick labels to the top and right
ax.tick_params(
    top=True,
    labeltop=True,
    bottom=False,
    labelbottom=False,
    right=True,
    labelright=True,
    left=False,
    labelleft=False,
)

# Move axis labels
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position("right")

ax.set_xlabel("X label")
ax.set_ylabel("Y label")

plt.show()
