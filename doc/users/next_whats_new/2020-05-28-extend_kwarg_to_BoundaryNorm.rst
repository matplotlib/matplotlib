New "extend" keyword to colors.BoundaryNorm
-------------------------------------------

`~.colors.BoundaryNorm` now has an ``extend`` kwarg,
analogous to ``extend`` in ~.axes._axes.Axes.contourf`. When set to
'both', 'min', or 'max', it interpolates such that the corresponding
out-of-range values are mapped to colors distinct from their in-range
neighbors.  The colorbar inherits the ``extend`` argument from the
norm, so with ``extend='both'``, for example, the colorbar will have
triangular extensions for out-of-range values with colors that differ
from adjacent colors.

  .. plot::

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import numpy as np

    # Make the data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    z = z[:-1, :-1]

    # Z roughly varies between -1 and +1.
    # Color boundary levels range from -0.8 to 0.8, so there are out-of-bounds
    # areas.
    levels = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    cmap = plt.get_cmap('PiYG')

    fig, axs = plt.subplots(nrows=2, constrained_layout=True, sharex=True)

    # Before this change:
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    im = axs[0].pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=axs[0], extend='both')
    axs[0].axis([x.min(), x.max(), y.min(), y.max()])
    axs[0].set_title("Colorbar with extend='both'")

    # With the new keyword:
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')
    im = axs[1].pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=axs[1])  # note that the colorbar is updated accordingly
    axs[1].axis([x.min(), x.max(), y.min(), y.max()])
    axs[1].set_title("BoundaryNorm with extend='both'")

    plt.show()
