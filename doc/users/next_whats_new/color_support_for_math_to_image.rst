``math_to_image`` now has a *color* keyword argument
--------------------------------------------------------

To easily support external libraries that rely on the MathText rendering of
Matplotlib to generate equation images, a *color* keyword argument was added
to `~matplotlib.mathtext.math_to_image`.

.. code-block:: python

    from matplotlib import mathtext
    mathtext.math_to_image('$x^2$', 'filename.png', color='Maroon')
