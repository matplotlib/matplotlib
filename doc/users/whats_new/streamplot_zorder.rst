Streamplot Zorder Keyword Argument Changes
------------------------------------------

The ``zorder`` parameter for :func:`streamplot` now has default
value of ``None`` instead of ``2``. If ``None`` is given as ``zorder``,
:func:`streamplot` has a default ``zorder`` of 
``matplotlib.lines.Line2D.zorder``.
