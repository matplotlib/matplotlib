``axes_grid1.axes_divider`` API changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``AxesLocator`` class is deprecated.  The ``new_locator`` method of divider
instances now instead returns an opaque callable (which can still be passed to
``ax.set_axes_locator``).

``Divider.locate`` is deprecated; use ``Divider.new_locator(...)(ax, renderer)``
instead.
