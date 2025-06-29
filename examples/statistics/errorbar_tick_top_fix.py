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
            rotation=55         # Rotate the tick labels for better visibility
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

    # Automatically hide tick labels that are not on the edge of the figure
    ax.label_outer()

# Adjust layout to prevent overlap between subplots and labels
plt.tight_layout()

# Display the plot
plt.show()
