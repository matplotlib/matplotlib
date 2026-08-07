Introduce open polygonal chains in ``matplotlib.widgets``
---------------------------------------------------------

The widget class `.widgets.PolygonSelector` is used to select a
polygon region of an Axes. More precisely, it enables the selection
of a closed polygonal chain, a sequence of vertices connected by
line segments where the first and last vertices are connected.

.. plot::
    :include-source: true
    :alt: Closed polygonal chains using PolygonSelector.

    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    _, ax = plt.subplots()
    selector = PolygonSelector(ax)
    selector.verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]

    plt.show()

Support for open polygonal chains has been added through the new
`.widgets.PolylineSelector` class. A polyline is a sequence of
vertices connected by line segments, where the first and
last vertices are not connected, i.e., an open polygonal chain.

.. plot::
    :include-source: true
    :alt: Open polygonal chains using PolylineSelector.

    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolylineSelector

    _, ax = plt.subplots()
    selector = PolylineSelector(ax)
    selector.verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]

    plt.show()

Both selectors share the same interactive editing capabilities,
including vertex repositioning and removal, as well as the ability
to define polygonal chains both programmatically and interactively.

The interactive selection of an open polygonal chain is completed by
pressing the *Enter* key after placing the final vertex.

Internally, the common functionality of polygonal chain selector
widgets has been extracted into the new private base class
`.widgets._PolygonalSelector`. `~.widgets.PolygonSelector` now
inherits from `~.widgets._PolygonalSelector` while preserving
its existing API and behavior.
