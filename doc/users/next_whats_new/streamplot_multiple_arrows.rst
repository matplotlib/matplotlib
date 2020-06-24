Multiple arrows on a streamline
-------------------------------

A new ``narrows`` argument has been added to `~matplotlib.axes.Axes.streamplot` that
allows more than one arrow to be added to each streamline:

.. plot::
    :include-source: true
    :alt: One charts, identified as ax and ax2, showing a streamplot. Each streamline
        has a three arrows.

    import matplotlib.pyplot as plt
    import numpy as np

    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2

    fig, ax = plt.subplots()
    ax.streamplot(X, Y, U, V, narrows=3)

    plt.show()
