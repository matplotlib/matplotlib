Stackplot styling
-----------------

`~.Axes.stackplot` now accepts sequences for the style parameters *facecolor*,
*edgecolor*, *linestyle*, and *linewidth*, similar to how the *hatch* parameter
is already handled.

.. plot::
    :include-source: true
    :alt: A stackplot showing with two regions.  The bottom region is red with black dots and a dotted black outline.  The top region is blue with gray stars and a thicker dashed outline.

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1

    fig, ax = plt.subplots()

    ax.stackplot(x, y1, y2,
                 facecolor=['tab:red', 'tab:blue'],
                 edgecolor=['black', 'gray'],
                 linestyle=[':', '--'],
                 linewidth=[2, 3],
                 hatch=['.', '*'])
