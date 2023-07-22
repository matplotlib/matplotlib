Clipping for contour plots
--------------------------

`~.Axes.contour` and `~.Axes.contourf` now accept the *clip_path* parameter.

.. plot::
    :include-source: true

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    x = y = np.arange(-3.0, 3.01, 0.025)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots()
    patch = mpatches.RegularPolygon((0, 0), 5, radius=2,
                                    transform=ax.transData)
    ax.contourf(X, Y, Z, clip_path=patch)

    plt.show()
