New details item to ``Line2D.contains()`` output
------------------------------------------------

A new item, ``'is_vert'``, is added to the ``details`` output dict of
`~.lines.Line2D.contains()` to complement the existing ``'ind'`` item.
The ``'is_vert'`` item is a boolean array of the same size as ``'ind'`` and
indicates whether the corresponding index in ``'ind'`` is an index of a vertex
of a line (``True``) or of an edge of a line (``False``)
