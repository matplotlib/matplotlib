Colormap method for creating a different lookup table size
----------------------------------------------------------
The new method `.Colormap.resampled` creates a new `.Colormap` instance
with the specified lookup table size. This is a replacement for manipulating
the lookup table size via ``get_cmap``.

Use::

    get_cmap(name).resampled(N)

instead of::

    get_cmap(name, lut=N)
