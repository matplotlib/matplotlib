import matplotlib.pyplot as plt
import numpy as np


def custom_bar_plot(labels, values, colors=None):
    """
    Custom bar plot function that plots bars with optional custom colors.

    Parameters:
        labels (array-like): Labels for each bar.
        values (array-like): Values for each bar.
        colors (array-like, optional): Colors for each bar. If not provided, default colors are used.

    Returns:
        None
    """
    x = np.arange(len(labels))
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    plt.bar(x, values, color=colors)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Custom Bar Plot')
    plt.xticks(x, labels)
    plt.grid(True)
    plt.show()
