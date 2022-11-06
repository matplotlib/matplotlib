View current appearance settings for ticks, tick labels, and gridlines
----------------------------------------------------------------------

The new `~matplotlib.axis.Axis.get_tick_params` method can be used to
retrieve the appearance settings that will be applied to any
additional ticks, tick labels, and gridlines added to the plot:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.yaxis.set_tick_params(labelsize=30, labelcolor='red',
                             direction='out', which='major')
    print(ax.yaxis.get_tick_params(which='major'))
    print(ax.yaxis.get_tick_params(which='minor'))
