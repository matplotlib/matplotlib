``PolygonSelector`` now supports open polygonal chains
------------------------------------------------------

The widget class `.widgets.PolygonSelector` is used to select a
polygonal region within an Axes. More precisely, it enables the
selection of a closed polygonal chain, i.e., a sequence of vertices
connected by line segments where the first and last vertices are connected.
These chains may be non-self-intersecting or self-intersecting.

.. plot::
    :include-source: true
    :alt: Closed polygonal chains using PolygonSelector.

    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    def setup(ax, title):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_title(title)
        ax.grid(alpha=0.5)

    fig, axs = plt.subplots(1, 2, layout="constrained")
    fig.suptitle("Closed polygonal chains")

    setup(axs[0], "Non-self-intersecting")
    selector1 = PolygonSelector(axs[0])
    selector1.verts = [(0.3, 0.4), (0.5, 0.2), (0.7, 0.4),
                       (0.7, 0.6), (0.5, 0.8), (0.3, 0.6)]

    setup(axs[1], "Self-intersecting")
    selector2 = PolygonSelector(axs[1])
    selector2.verts = [(0.2, 0.5), (0.8, 0.4), (0.6, 0.2), (0.5, 0.8)]

    plt.show()

An open polygonal chain is a sequence of vertices connected by line
segments, where the first and last vertices are not connected.
`~.widgets.PolygonSelector` now supports both programmatic and
interactive selection of open polygonal chains within an Axes,
including both non-self-intersecting and self-intersecting chains.

.. plot::
    :include-source: true
    :alt: Open polygonal chains using PolygonSelector.

    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    def setup(ax, title):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_title(title)
        ax.grid(alpha=0.5)

    fig, axs = plt.subplots(1, 2, layout="constrained")
    fig.suptitle("Open polygonal chains")

    setup(axs[0], "Non-self-intersecting")
    selector1 = PolygonSelector(axs[0], closed=False)
    selector1.verts = [(0.3, 0.2), (0.5, 0.3), (0.6, 0.4), (0.7, 0.6), (0.7, 0.7),
                       (0.6, 0.8), (0.5, 0.8), (0.4, 0.7), (0.4, 0.6), (0.5, 0.5)]

    setup(axs[1], "Self-intersecting")
    selector2 = PolygonSelector(axs[1], closed=False)
    selector2.verts = [(0.2, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.7), (0.5, 0.6),
                       (0.6, 0.5), (0.7, 0.5), (0.8, 0.6), (0.8, 0.7), (0.7, 0.8),
                       (0.6, 0.8), (0.5, 0.7), (0.4, 0.5), (0.4, 0.3), (0.5, 0.2)]

    plt.show()

A new parameter, ``closed``, has been added to `~.widgets.PolygonSelector`
to explicitly control whether the polygonal chain is open or closed.
By default, the parameter selects closed polygonal regions, ``closed=True``,
preserving the current behavior.

The interactive selection of an open polygonal chain is completed by
pressing the Enter key after placing a vertex. The existing interactive
editing functionality is preserved and applies consistently to both modes.
