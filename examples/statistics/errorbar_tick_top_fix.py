"""
Workaround for `tick_top()` with `sharex=True` subplots

Using `tick_top()` on subplots with `sharex=True` does not restore tick labels
on the top-most axes because shared axes suppress redundant labels.

This example demonstrates a workaround using `tick_params()` and `label_outer()`
to explicitly enable top ticks and labels on the first row of a subplot grid.
"""

import matplotlib.pyplot as plt

# Create a 2x2 grid of subplots with shared x and y axes
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    sharex=True, sharey=True
)

# Flatten the 2D array of axes for easier iteration
axes = axes.flatten()

for i, ax in enumerate(axes):
    # Plot a basic scatter plot
    ax.scatter(x=[0, 1, 5, 3], y=[0, 1, 2, 4])

    # Apply ticks and labels to the top of the axes only for the top row
    if i < 2:
        ax.tick_params(
            axis='x',
            top=True,           # Show ticks on top
            labeltop=True,      # Show labels on top
            bottom=False,       # Hide bottom ticks
            labelbottom=False,  # Hide bottom labels
            rotation=55         # Rotate tick labels
        )
    else:
        # Hide all x-axis ticks/labels for the bottom row
        ax.tick_params(
            axis='x',
            top=False,
            labeltop=False,
            bottom=False,
            labelbottom=False
        )

    # Automatically hide labels on inner axes to avoid redundancy
    ax.label_outer()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
