``offset`` parameter for MultipleLocator
----------------------------------------

An *offset* may now be specified to shift all the ticks by the given value.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _, ax = plt.subplots()
    ax.plot(range(10))
    locator = mticker.MultipleLocator(base=3, offset=0.3)
    ax.xaxis.set_major_locator(locator)

    plt.show()
