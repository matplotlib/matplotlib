Added center and frame kwargs to pie
````````````````````````````````````

These control where the center of the pie graph are and if
the Axes frame is shown.

Contour plot corner masking
```````````````````````````

Ian Thomas rewrote the C++ code that calculates contours to add support for
corner masking.  This is controlled by a new keyword argument
``corner_mask`` in the functions :func:`~matplotlib.pyplot.contour` and
:func:`~matplotlib.pyplot.contourf`.  The previous behaviour, which is now
obtained using ``corner_mask=False``, was for a single masked point to
completely mask out all four quads touching that point.  The new behaviour,
obtained using ``corner_mask=True``, only masks the corners of those
quads touching the point; any triangular corners comprising three unmasked
points are contoured as usual.  If the ``corner_mask`` keyword argument is not
specified, the default value is taken from rcParams.

.. plot:: mpl_examples/pylab_examples/contour_corner_mask.py
