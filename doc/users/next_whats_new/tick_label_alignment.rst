Added tick label alignment parameters to tick_params
----------------------------------------

The ``tick_params`` method now supports setting the horizontal and vertical alignment
of tick labels using the ``labelhorizontalalignment`` and ``labelverticalalignment``
parameters. These parameters can be used when specifying a single axis ('x' or 'y')
to control the alignment of tick labels:

.. code-block:: python

    ax.tick_params(axis='x', labelhorizontalalignment='right')
    ax.tick_params(axis='y', labelverticalalignment='top')

This provides more control over tick label positioning and can be useful for
improving the readability and appearance of plots. 