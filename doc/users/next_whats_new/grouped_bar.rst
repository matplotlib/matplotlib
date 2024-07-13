Grouped bar charts
------------------

The new method `~.Axes.grouped_bar()` simplifies the creation of grouped bar charts
significantly. It supports different input data types (lists of datasets, dicts of
datasets, data in 2D arrays, pandas DataFrames), and allows for easy customization
of placement via controllable distances between bars and between bar groups.

Example:

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt

    categories = ['A', 'B']
    datasets = {
        'dataset 0': [1.0, 3.0],
        'dataset 1': [1.4, 3.4],
        'dataset 2': [1.8, 3.8],
    }

    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.grouped_bar(datasets, tick_labels=categories)
    ax.legend()
