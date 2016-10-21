Code Removal
````````````

matplotlib.delaunay
-------------------
Remove the delaunay triangulation code which is now handled by Qhull
via ``matplotlib.tri``


qt4_compat.py
-------------
Moved to ``qt_compat.py``.  Renamed because it now handles Qt5 as well.


Deprecated methods
------------------

The ``GraphicsContextBase.set_graylevel``, ``FigureCanvasBase.onHilite`` and
``mpl_toolkits.axes_grid1.mpl_axes.Axes.toggle_axisline`` methods have been
removed.


`Axes.set_aspect("normal")`
---------------------------

Support for setting an ``Axes``' aspect to ``"normal"`` has been removed, in
favor of the synonym ``"auto"``.
