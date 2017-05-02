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

    from matplotlib import pyplot as plt
    import matplotlib as mpl

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.7, 0.9, 0.2])
    ax2 = fig.add_axes([0.05, 0.2, 0.9, 0.2])

    # Set the colormap and bounds
    bounds = [-1, 2, 5, 7, 12, 15]
    cmap = mpl.cm.get_cmap('viridis')

    # Default behavior
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                         norm=norm,
                                         extend='both',
                                         orientation='horizontal')
    cb1.set_label('Default BoundaryNorm ouput')

    # New behavior
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                         norm=norm,
                                         orientation='horizontal')
    cb2.set_label("With new extend='both' keyword")

    plt.show()