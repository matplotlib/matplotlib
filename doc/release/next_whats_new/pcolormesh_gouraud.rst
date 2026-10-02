``gouraud`` shading supports data at quadrilaterals centers
-----------------------------------------------------------

`~.Axes.pcolormesh` previously required data at the corners of
quadrilaterals for ``gouraud`` shading. It now also supports data
defined at the centers of quadrilaterals.

This now allows `~.Axes.pcolormesh` to provide both constant and
linearly interpolated shading for each data location.

=============== ======== =====================
..              constant linearly interpolated
=============== ======== =====================
data at corners nearest  gouraud
data at centers flat     gouraud
=============== ======== =====================

Shading can be switched for the same data location, allowing
switches between ``nearest`` and ``gouraud``, and now also
between ``flat`` and ``gouraud``.

For example:

.. plot::
    :include-source: true
    :alt: Switching between constant and linearly interpolated shading for data at corners and centers.

    import matplotlib.pyplot as plt
    import numpy as np

    nrows, ncols = 3, 5
    Z = np.arange(nrows * ncols).reshape(nrows, ncols)
    x = np.arange(ncols + 1)
    y = np.arange(nrows + 1)

    fig, axs = plt.subplots(2, 2, layout='constrained')

    # Data at corners, requires X and Y the same shape as Z.
    axs[0, 0].pcolormesh(x[:-1], y[:-1], Z, shading='nearest')
    axs[0, 0].set_title('nearest: X, Y, Z same shape')

    axs[0, 1].pcolormesh(x[:-1], y[:-1], Z, shading='gouraud')
    axs[0, 1].set_title('gouraud: X, Y, Z same shape')

    # Data at centers, requires X and Y one larger than Z.
    axs[1, 0].pcolormesh(x, y, Z, shading='flat')
    axs[1, 0].set_title('flat: X, Y one larger than Z')

    axs[1, 1].pcolormesh(x, y, Z, shading='gouraud')
    axs[1, 1].set_title('gouraud: X, Y one larger than Z')

    plt.show()
