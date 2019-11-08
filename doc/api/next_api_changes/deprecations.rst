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

``den`` parameter and attribute to :mod:`mpl_toolkits.axisartist.angle_helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For all locator classes defined in :mod:`mpl_toolkits.axisartist.angle_helper`,
the ``den`` parameter has been renamed to ``nbins``, and the ``den`` attribute
deprecated in favor of its (preexisting) synonym ``nbins``, for consistency
with locator classes defined in :mod:`matplotlib.ticker`.

``backend_pgf.LatexManager.latex_stdin_utf8``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``backend_pgf.LatexManager.latex`` is now created with ``encoding="utf-8"``, so
its ``stdin`` attribute is already utf8-encoded; the ``latex_stdin_utf8``
attribute is thus deprecated.

Flags containing "U" passed to `.cbook.to_filehandle` and `.cbook.open_file_cm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please remove "U" from flags passed to `.cbook.to_filehandle` and
`.cbook.open_file_cm`.  This is consistent with their removal from `open` in
Python 3.9.
