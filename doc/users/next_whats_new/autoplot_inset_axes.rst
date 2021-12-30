Addition of an inset Axes with automatic zoom plotting
------------------------------------------------------

It is now possible to create an inset axes that is a zoom-in on a region in
the parent axes without needing to replot all items a second time, using the
`~matplotlib.axes.Axes.inset_zoom_axes` method of the
`~matplotlib.axes.Axes` class. Arguments for this method are backwards
compatible with the `~matplotlib.axes.Axes.inset_axes` method.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(1)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot([i for i in range(10)], "r-o")
    ax.text(3, 2.5, "Hello World!", ha="center")
    ax.imshow(np.random.rand(30, 30), origin="lower", cmap="Blues", alpha=0.5)
    axins = ax.inset_zoom_axes([0.5, 0.5, 0.48, 0.48])
    axins.set_xlim(1, 5)
    axins.set_ylim(1, 5)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.show()
