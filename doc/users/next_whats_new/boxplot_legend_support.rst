Legend support for Boxplot
~~~~~~~~~~~~~~~~~~~~~~~~~~
Boxplots now support a *label* parameter to create legend entries.

Legend labels can be passed as a list of strings to label multiple boxes in a single
`.Axes.boxplot` call:


.. plot::
    :include-source: true
    :alt: Example of creating 3 boxplots and assigning legend labels as a sequence.

    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(19680801)
    fruit_weights = [
        np.random.normal(130, 10, size=100),
        np.random.normal(125, 20, size=100),
        np.random.normal(120, 30, size=100),
    ]
    labels = ['peaches', 'oranges', 'tomatoes']
    colors = ['peachpuff', 'orange', 'tomato']

    fig, ax = plt.subplots()
    ax.set_ylabel('fruit weight (g)')

    bplot = ax.boxplot(fruit_weights,
                       patch_artist=True,  # fill with color
                       label=labels)

    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks([])
    ax.legend()


Or as a single string to each individual `.Axes.boxplot`:

.. plot::
    :include-source: true
    :alt: Example of creating 2 boxplots and assigning each legend label as a string.

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    data_A = np.random.random((100, 3))
    data_B = np.random.random((100, 3)) + 0.2
    pos = np.arange(3)

    ax.boxplot(data_A, positions=pos - 0.2, patch_artist=True, label='Box A',
               boxprops={'facecolor': 'steelblue'})
    ax.boxplot(data_B, positions=pos + 0.2, patch_artist=True, label='Box B',
               boxprops={'facecolor': 'lightblue'})

    ax.legend()
