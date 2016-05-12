Improved color conversion API and RGBA support
----------------------------------------------

The :module:`~matplotlib.colors` gained a new color conversion API with
full support for the alpha channel.  The main public functions are
:func:`~matplotlib.colors.is_color_like`, :func:`matplotlib.colors.to_rgba`,
:func:`matplotlib.colors.to_rgba_array` and :func:`~matplotlib.colors.to_hex`.
RGBA quadruplets are encoded in hex format as `#rrggbbaa`.

A side benefit is that the Qt options editor now allows setting the alpha
channel of the artists as well.
