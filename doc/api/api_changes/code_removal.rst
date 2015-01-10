Code Removal
````````````

matplotlib.delaunay
-------------------
Remove the delaunay triangulation code which is now handled by Qhull
via ``matplotlib.tri``


qt4_compat.py
-------------
Moved to ``qt_compat.py``.  Renamed because it now handles Qt5 as well.
