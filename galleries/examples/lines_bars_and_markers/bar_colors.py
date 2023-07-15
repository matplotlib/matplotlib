"""
==============
Bar color demo
==============

This is an example showing how to control bar color and legend entries
using the *color* and *label* parameters of `~matplotlib.pyplot.bar`.
Note that labels with a preceding underscore won't show up in the legend.
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')

plt.show()


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


# Example usage
labels = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
values = [15, 8, 12, 10, 5]
colors = ['red', 'blue', 'green', 'orange', 'purple']

custom_bar_plot(labels, values, colors)
