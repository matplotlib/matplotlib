Deprecations
------------

``figure.add_axes()`` without arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calling ``fig.add_axes()`` with no arguments currently does nothing. This call
will raise an error in the future. Adding a free-floating axes needs a position
rectangle. If you want a figure-filling single axes, use ``add_subplot()``
instead.

``backend_wx.DEBUG_MSG``
~~~~~~~~~~~~~~~~~~~~~~~~
``backend_wx.DEBUG_MSG`` is deprecated.  The wx backends now use regular
logging.

``Colorbar.config_axis()``
~~~~~~~~~~~~~~~~~~~~~~~~~~
``Colorbar.config_axis()`` is considered internal. Its use is deprecated.

``NonUniformImage.is_grayscale`` and ``PcolorImage.is_grayscale``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated, for consistency with ``AxesImage.is_grayscale``,
which was removed back in Matplotlib 2.0.0.  (Note that previously, these
attributes were only available *after rendering the image*).
