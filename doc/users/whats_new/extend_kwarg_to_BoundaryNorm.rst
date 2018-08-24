New "extend" keyword to colors.BoundaryNorm
-------------------------------------------

:func:`~matplotlib.colors.BoundaryNorm` now has an ``extend`` kwarg. This is
useful when creating a discrete colorbar from a continuous colormap: when
setting ``extend`` to ``'both'``, ``'min'`` or ``'max'``, the colors are
interpolated so that the extensions have a different color than the inner
cells.

Example
```````
::

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import numpy as np

    # Make the data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    z = z[:-1, :-1]

    # Z roughly varies between -1 and +1
    # my levels are chosen so that the color bar should be extended
    levels = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    cmap = plt.get_cmap('PiYG')

    # Before this change
    plt.subplot(2, 1, 1)
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    plt.colorbar(extend='both')
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.title('pcolormesh with extended colorbar')

    # With the new keyword
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')
    plt.subplot(2, 1, 2)
    im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    plt.colorbar()  # note that the colorbar is updated accordingly
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.title('pcolormesh with extended BoundaryNorm')

    plt.show()
