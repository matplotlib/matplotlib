Seperate styling options for major/minor and x/y axis grid line in rcParams
---------------------------------------------------------------------------

Using :rc:`grid.major.*` or :rc:`grid.minor.*` will overwrite the value in
:rc:`grid.*` for the major and minor gridlines, respectively. Similarly,
specifying :rc:`grid.xaxis.major.*` and :rc:`grid.yaxis.major.*` will overwrite
`grid.major.*` for x and y axis major gridlines respectively.

.. plot::
    :include-source: true
    :alt: Modifying the gridlines for three figures using the new options `rcParams`

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Using old values
    mpl.rcdefaults()
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["axes.grid.which"] = "both"
    mpl.rcParams["grid.color"] = "red"
    mpl.rcParams["grid.linewidth"] = 1
    mpl.rcParams["grid.linestyle"] = "-"
    mpl.rcParams["grid.alpha"] = 1
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("Runtime both set implicitly")

    # Overwriting major and minor settings
    mpl.rcdefaults()
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["axes.grid.which"] = "both"
    mpl.rcParams["grid.color"] = "red"
    mpl.rcParams["grid.linewidth"] = 1
    mpl.rcParams["grid.linestyle"] = "-"
    mpl.rcParams["grid.alpha"] = 1

    mpl.rcParams["grid.major.color"] = "black"
    mpl.rcParams["grid.major.linewidth"] = 2
    mpl.rcParams["grid.major.linestyle"] = ":"
    mpl.rcParams["grid.minor.color"] = "gray"
    mpl.rcParams["grid.minor.linewidth"] = 0.5
    mpl.rcParams["grid.minor.linestyle"] = "--"
    mpl.rcParams["grid.minor.alpha"] = 0.5
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("Runtime explicitly set major and minor")

    # Overwriting x and y axis for majro and minor lines
    mpl.rcdefaults()
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["axes.grid.which"] = "both"
    mpl.rcParams["grid.color"] = "red"
    mpl.rcParams["grid.linewidth"] = 1
    mpl.rcParams["grid.linestyle"] = "-"
    mpl.rcParams["grid.alpha"] = 1

    mpl.rcParams["grid.major.color"] = "black"
    mpl.rcParams["grid.major.linewidth"] = 2
    mpl.rcParams["grid.major.linestyle"] = ":"
    mpl.rcParams["grid.minor.color"] = "gray"
    mpl.rcParams["grid.minor.linewidth"] = 0.5
    mpl.rcParams["grid.minor.linestyle"] = "--"
    mpl.rcParams["grid.minor.alpha"] = 0.5

    mpl.rcParams["grid.xaxis.major.color"] = "red"
    mpl.rcParams["grid.xaxis.major.linewidth"] = 5
    mpl.rcParams["grid.xaxis.major.alpha"] = 0.2
    mpl.rcParams["grid.yaxis.minor.linestyle"] = "-"
    mpl.rcParams["grid.yaxis.major.linewidth"] = 3
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("Runtime explicitly set some values for x and y axis")
    
    plt.show()
