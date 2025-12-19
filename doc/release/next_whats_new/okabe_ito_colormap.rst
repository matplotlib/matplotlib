Okabe-Ito accessible color sequence
-----------------------------------

Matplotlib now includes the Okabe-Ito colormap. This addition provides more accessibility for categorical color needs and makes it easier to produce figures that are readable for viewers with common forms of color vision deficiency. This color sequence is unambiguous regardless of whether the viewer has colorblindness and is reliably print-friendly.

You can use the sequence anywhere a listed colormap or color cycle is accepted. It is available alongside other qualitative color sets as ``okabe_ito``.

For example, to set it as the default colormap for your plots and image-like artists, use:

.. code-block:: python

    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams['axes.prop_cycle'] = cycler('color', plt.colormaps['okabe_ito'].colors)
    plt.rcParams['image.cmap'] = 'okabe_ito'

Or, when creating plots, you can pass the colormap explicitly:

.. plot::

    import matplotlib.pyplot as plt

    colors = plt.colormaps['okabe_ito'].colors
    x = range(5)
    for i, c in enumerate(colors):
        plt.plot(x, [v*(i+1) for v in x], color=c, label=f'line {i}')
    plt.legend()
    plt.show()
