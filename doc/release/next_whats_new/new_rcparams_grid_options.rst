Separate styling options for major/minor grid line in rcParams
--------------------------------------------------------------

Using :rc:`grid.major.*` or :rc:`grid.minor.*` will overwrite the value in
:rc:`grid.*` for the major and minor gridlines, respectively.

.. plot::
    :include-source: true
    :alt: Modifying the gridlines using the new options `rcParams`

    import matplotlib as mpl
    import matplotlib.pyplot as plt


    # Set visibility for major and minor gridlines
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["axes.grid.which"] = "both"

    # Using old values to set both major and minor properties
    mpl.rcParams["grid.color"] = "red"
    mpl.rcParams["grid.linewidth"] = 1

    # Overwrite some values for major and minor separately
    mpl.rcParams["grid.major.color"] = "black"
    mpl.rcParams["grid.major.linewidth"] = 2
    mpl.rcParams["grid.minor.linestyle"] = ":"
    mpl.rcParams["grid.minor.alpha"] = 0.6

    plt.plot([0, 1], [0, 1])

    plt.show()
