``allsegs``, ``allkinds``, ``tcolors`` and ``tlinewidths`` attributes of `.ContourSet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated; if required, directly retrieve the vertices
and codes of the Path objects from ``ContourSet.get_paths()`` and the colors
and the linewidths via ``ContourSet.get_facecolor()``, ``ContourSet.get_edgecolor()``
and ``ContourSet.get_linewidths()``.
