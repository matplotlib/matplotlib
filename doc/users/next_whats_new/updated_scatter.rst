Enable configuration of empty markers in `~matplotlib.axes.Axes.scatter`
------------------------------------------------------------------------

`~matplotlib.axes.Axes.scatter` can now be configured to plot empty markers by setting ``facecolors`` to *'none'* and defining ``c``. In this case, ``c`` will be now used as ``edgecolor``.

.. plot::
    :include-source: true
    :alt: A simple scatter plot which illustrates the use of the *scatter* function to plot empty markers.

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 10)
    plt.scatter(x, x, c=x, facecolors='none', marker='o')
    plt.show()
