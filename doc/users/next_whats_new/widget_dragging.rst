Dragging selectors
------------------

The `~matplotlib.widgets.RectangleSelector` and
`~matplotlib.widgets.EllipseSelector` have a new keyword argument,
*drag_from_anywhere*, which when set to `True` allows you to click and drag
from anywhere inside the selector to move it. Previously it was only possible
to move it by either activating the move modifier button, or clicking on the
central handle.
