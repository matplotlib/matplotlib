``violinplot`` and ``violin`` orientation parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Violinplots have a new parameter *orientation: {"vertical", "horizontal"}*
to change the orientation of the plot. This will replace the deprecated
*vert: bool* parameter.


.. plot::
    :include-source: true
    :alt: Example of creating 4 horizontal violinplots.

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    ax.violinplot(all_data, orientation='horizontal')
    plt.show()
