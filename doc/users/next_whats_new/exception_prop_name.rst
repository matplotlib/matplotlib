Exception handling control
~~~~~~~~~~~~~~~~~~~~~~~~~~

The exception raised when an invalid keyword parameter is passed now includes
that parameter name as the exception's ``name`` property.  This provides more
control for exception handling:


.. code-block:: python

    import matplotlib.pyplot as plt

    def wobbly_plot(args, **kwargs):
        w = kwargs.pop('wobble_factor', None)

        try:
            plt.plot(args, **kwargs)
        except AttributeError as e:
            raise AttributeError(f'wobbly_plot does not take parameter {e.name}') from e


    wobbly_plot([0, 1], wibble_factor=5)

.. code-block::

    AttributeError: wobbly_plot does not take parameter wibble_factor
