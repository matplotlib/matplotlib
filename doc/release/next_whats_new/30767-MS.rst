``legend.linewidth`` rcParam and parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new rcParam ``legend.linewidth`` has been added to control the line width of
the legend's box edges. When set to ``None`` (the default), it inherits the
value from ``patch.linewidth``. This allows for independent control of the
legend frame line width without affecting other elements.

The `.Legend` constructor also accepts a new *linewidth* parameter to set the
legend frame line width directly, overriding the rcParam value.

.. plot::
    :include-source: true
    :alt: A line plot with a legend showing a thick border around the legend box.

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], label='data')
    ax.legend(linewidth=2.0)  # Thick legend box edge
    plt.show()
