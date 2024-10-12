``boxplot`` and ``bxp`` orientation parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boxplots have a new parameter *orientation: {"vertical", "horizontal"}*
to change the orientation of the plot. This replaces the deprecated
*vert: bool* parameter.


.. plot::
    :include-source: true
    :alt: Example of creating 4 horizontal boxplots.

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    ax.boxplot(all_data, orientation='horizontal')
    plt.show()
