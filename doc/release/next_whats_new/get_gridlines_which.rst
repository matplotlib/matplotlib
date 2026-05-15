``Axis.get_gridlines`` can return minor gridlines
-------------------------------------------------
`~matplotlib.axis.Axis.get_gridlines` now accepts a *which* keyword argument
to select major, minor, or both groups of gridlines. The default value
``'major'`` preserves the previous behavior.

.. plot::
    :include-source: true
    :alt: Highlight every minor gridline of the x-axis in red.

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(range(10))
    ax.minorticks_on()
    ax.grid(which='both')

    for line in ax.xaxis.get_gridlines(which='minor'):
        line.set_color('red')

    plt.show()

Previously there was no public API to access minor gridlines, so downstream
libraries reached into the private ``Axis._minor_tick_kw`` mapping to detect
their state.
