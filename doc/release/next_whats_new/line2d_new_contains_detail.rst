New details item in ``Line2D.contains()`` output
------------------------------------------------

A new item, ``'vertex_hit'``, is added to the ``details`` output dict of
`lines.Line2D.contains()` to complement the existing ``'ind'`` item.
The ``'vertex_hit'`` item is a boolean array of the same size as ``'ind'`` and
indicates whether the corresponding index in ``'ind'`` is an index of a vertex
of the line (``True``)
