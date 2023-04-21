Enable configuration of empty markers in `~matplotlib.axes.Axes.scatter`
------------------------------------------------------------------------

`~matplotlib.axes.Axes.scatter` can now be configured to plot empty markers 
without additional code. Setting ``facecolors`` to *'none'* and
defining ``c`` now draws only the edge colors for fillable markers. This
allows empty face colors and non-empty edge colors without disrupting
existing color-mapping capabilities.

.. plot::
    :include-source: true
    :alt: A simple scatter plot which illustrates the use of the *scatter* function to plot empty markers.

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 10)
    plt.scatter(x, x, c=x, facecolors='none', marker='o')
    plt.show()
