Rotating selectors
------------------

The `~matplotlib.widgets.RectangleSelector` and
`~matplotlib.widgets.EllipseSelector` can now be rotated by pressing the 'r'
key, and dragging one of their control points. This is currently only
implemented for equal-aspect axes. The rotation is done about the
first vertex placed when drawing the selector.
