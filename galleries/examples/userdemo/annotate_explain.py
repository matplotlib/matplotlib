"""
================
Annotate Explain
================

"""

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

# Create a figure with four subplots arranged in a 2x2 grid
fig, axs = plt.subplots(2, 2)

# Define the x and y coordinates of two points
x1, y1 = 0.3, 0.3
x2, y2 = 0.7, 0.7

# First subplot: connect arrow style
ax = axs.flat[0]
ax.plot([x1, x2], [y1, y2], ".")    # plot the two points
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)    # create an ellipse patch
ax.add_artist(el)    # add the ellipse patch to the plot

# add an arrow connecting the two points with the connect style
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchB=None,    # Do not use any patch
                            shrinkB=0,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )

# add a text label to the subplot
ax.text(.05, .95, "connect", transform=ax.transAxes, ha="left", va="top")

# Second subplot: clip arrow style
ax = axs.flat[1]
ax.plot([x1, x2], [y1, y2], ".")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)

# add an arrow connecting the two points with the clip style
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchB=el,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )
ax.text(.05, .95, "clip", transform=ax.transAxes, ha="left", va="top")

# Third subplot: shrink arrow style
ax = axs.flat[2]
ax.plot([x1, x2], [y1, y2], ".")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)

# add an arrow connecting the two points with the shrink style
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchB=el,
                            shrinkB=5,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )
ax.text(.05, .95, "shrink", transform=ax.transAxes, ha="left", va="top")

# Fourth subplot: fancy arrow style
ax = axs.flat[3]
ax.plot([x1, x2], [y1, y2], ".")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)

# Add an arrow connecting the two points with the fancy style
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="fancy",
                            color="0.5",
                            patchB=el,
                            shrinkB=5,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )

# Add text to the plot to label the arrow as "connect"
ax.text(.05, .95, "mutate", transform=ax.transAxes, ha="left", va="top")

# Set the limits of the plot and remove the tick marks
for ax in axs.flat:
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect=1)

# Show the plot
plt.show()
