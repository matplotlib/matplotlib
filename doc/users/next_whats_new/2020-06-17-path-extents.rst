New function to get Path's *stroked* Bbox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Path is typically drawn by stroking it (with some ``markeredgewidth``), an
operation which changes its bounding box in a nontrivial way, depending on the
Path's joinstyle, capstyle, miterlimit, and shape.

`~.path.Path.get_stroked_extents` was added to allow computation of the final
bounding box in pixel/points coordinates of the line, after it has been drawn
(and accounting for the joinstyle, capstyle, and miterlimit).
