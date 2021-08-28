Colormap registry (experimental)
--------------------------------

Colormaps are now managed via `matplotlib.colormaps` (or `.pyplot.colormaps`),
which is a `.ColormapRegistry`. While we are confident that the API is final,
we formally mark it as experimental for 3.5 because we want to keep the option
to still adapt the API for 3.6 should the need arise.

Colormaps can be obtained using item access::

    import matplotlib.pyplot as plt
    cmap = plt.colormaps['viridis']

To register new colormaps use::

    plt.colormaps.register(my_colormap)

We recommend to use the new API instead of the `~.cm.get_cmap` and
`~.cm.register_cmap` functions for new code. `matplotlib.cm.get_cmap` and
`matplotlib.cm.register_cmap` will eventually be deprecated and removed.
Within `.pyplot` ``plt.get_cmap()`` and ``plt.register_cmap()`` will continue
to be supported for backward compatibility.