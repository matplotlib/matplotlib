Get/set window title methods have been removed from the canvas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the corresponding methods on the FigureManager if using pyplot,
or GUI-specific methods if embedding.

``ContourLabeler.get_label_coords()`` has been removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no replacement, it was considered an internal helper.

The **return_all** keyword argument has been removed from ``gridspec.get_position()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **minimum_descent** has been removed from ``TextArea``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The minimum_descent is now effectively always True.

Extra parameters to Axes constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters of the Axes constructor other than *fig* and *rect* are now keyword only.

``sphinext.plot_directive.align`` has been removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``docutils.parsers.rst.directives.images.Image.align`` instead.
