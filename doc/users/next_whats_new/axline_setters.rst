Axline setters and getters
--------------------------

The returned object from `.axes.Axes.axline` now supports getter and setter
methods for its *xy1*, *xy2* and *slope* attributes:

.. code-block:: python

    line1.get_xy1()
    line1.get_slope()
    line2.get_xy2()

.. code-block:: python

    line1.set_xy1(.2, .3)
    line1.set_slope(2.4)
    line2.set_xy2(.1, .6)
