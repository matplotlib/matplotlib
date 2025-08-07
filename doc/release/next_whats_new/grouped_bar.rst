Grouped bar charts
------------------

The new method `~.Axes.grouped_bar()` simplifies the creation of grouped bar charts
significantly. It supports different input data types (lists of datasets, dicts of
datasets, data in 2D arrays, pandas DataFrames), and allows for easy customization
of placement via controllable distances between bars and between bar groups.

Example:

.. plot::
    :include-source: true
    :alt: Diagram of a grouped bar chart of 3 datasets with 2 categories.

    import matplotlib.pyplot as plt

    categories = ['A', 'B']
    datasets = {
        'dataset 0': [1, 11],
        'dataset 1': [3, 13],
        'dataset 2': [5, 15],
    }

    fig, ax = plt.subplots()
    ax.grouped_bar(datasets, tick_labels=categories)
    ax.legend()
