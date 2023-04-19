``Figure.callbacks`` is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Figure ``callbacks`` property has been removed. The only signal was
"dpi_changed", which can be replaced by connecting to the "resize_event" on the
canvas ``figure.canvas.mpl_connect("resize_event", func)`` instead.



Passing too many positional arguments to ``tripcolor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... raises ``TypeError`` (extra arguments were previously ignored).


The *filled* argument to ``Colorbar`` is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This behavior was already governed by the underlying ``ScalarMappable``.


Widgets
~~~~~~~

The *visible* attribute setter of Selector widgets has been removed; use ``set_visible``
The associated getter is also deprecated, but not yet expired.
