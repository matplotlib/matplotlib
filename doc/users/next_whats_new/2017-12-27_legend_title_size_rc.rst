Legend Title Size rc parameter
------------------------------

A new rc parameter has been added as an option in matplotlibrc allowing you to explicitly control the font size of the legend title.
The default option is ``inherit`` which reverts to the previous behavior of inheriting font size from ``font.size``.

``legend.titlesize = 'large'``

.. code-block:: python

    import matplotlib.rcParams as rcParams

    rcParams['legend.titlesize'] = 'large'
