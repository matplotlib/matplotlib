Okabe-Ito accessible color sequence
-----------------------------------

Matplotlib now includes the `Okabe-Ito color sequence`_. Its colors remain distinguishable for common forms of color-vision deficiency and when printed.

.. _Okabe-Ito color sequence: https://jfly.uni-koeln.de/color/#pallet

For example, to set it as the default colormap for your plots and image-like artists, use:

.. code-block:: python

    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams['axes.prop_cycle'] = cycler('color', plt.colormaps['okabe_ito'].colors)
    plt.rcParams['image.cmap'] = 'okabe_ito'

Or, when creating plots, you can pass it explicitly:

.. plot::

    import matplotlib.pyplot as plt

    colors = plt.colormaps['okabe_ito'].colors
    x = range(5)
    for i, c in enumerate(colors):
        plt.plot(x, [v*(i+1) for v in x], color=c, label=f'line {i}')
    plt.legend()
    plt.show()
