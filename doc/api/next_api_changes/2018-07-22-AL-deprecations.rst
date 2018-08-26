Deprecations
------------

Support for custom backends that do not provide a ``set_hatch_color`` method is
deprecated.  We suggest that custom backends let their ``GraphicsContext``
class inherit from `GraphicsContextBase`, to at least provide stubs for all
required methods.

The fields ``Artist.aname`` and ``Axes.aname`` are deprecated. Please use
``isinstance()`` or ``__class__.__name__`` checks instead.