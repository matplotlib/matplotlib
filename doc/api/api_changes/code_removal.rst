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

The ``ArtistInspector.findobj`` method, which was never working due to the lack
of a ``get_children`` method, has been removed.

The deprecated ``point_in_path``, ``get_path_extents``,
``point_in_path_collection``, ``path_intersects_path``,
``convert_path_to_polygons``, ``cleanup_path`` and ``clip_path_to_rect``
functions in the ``matplotlib.path`` module have been removed.  Their
functionality remains exposed as methods on the ``Path`` class.


`Axes.set_aspect("normal")`
---------------------------

Support for setting an ``Axes``' aspect to ``"normal"`` has been removed, in
favor of the synonym ``"auto"``.


``shading`` kwarg to ``pcolor``
-------------------------------

The ``shading`` kwarg to ``pcolor`` has been removed.  Set ``edgecolors``
appropriately instead.


Removed internal functions
--------------------------

The ``matplotlib.backends.backend_ps.seq_allequal`` function has been removed.
Use ``np.array_equal`` instead.

The deprecated ``matplotlib.rcsetup.validate_maskedarray``,
``matplotlib.rcsetup.deprecate_savefig_extension`` and
``matplotlib.rcsetup.validate_tkpythoninspect`` functions, and associated
``savefig.extension`` and ``tk.pythoninspect`` rcparams entries have been
removed.
