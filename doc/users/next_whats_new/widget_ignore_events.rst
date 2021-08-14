Ignore events outside selection
-------------------------------
The `~matplotlib.widgets.SpanSelector`, `~matplotlib.widgets.RectangleSelector`
and `~matplotlib.widgets.EllipseSelector` have a new keyword argument,
*ignore_event_outside*, which when set to `True` will ignore events outside of
the current selection. The handles or the new dragging functionality can instead
be used to change the selection.
