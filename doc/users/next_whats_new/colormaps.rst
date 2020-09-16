Colormaps registry
------------------

Colormaps are now managed via `matplotlib.colormaps`, which is a
`.ColormapRegistry`.

Colormaps can be obtained using item access::

    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']

To register new colormaps use::

    mpl.colormaps.register(my_colormap)

The use of `matplotlib.cm.get_cmap` and `matplotlib.cm.register_cmap` is
discouraged in favor of the above. Within `.pyplot` the use of
``plt.get_cmap()`` and ``plt.register_cmap()`` will continue to be supported.