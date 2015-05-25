OffsetBoxes now support clipping
````````````````````````````````

`Artists` draw onto objects of type :class:`~OffsetBox`
through :class:`~OffsetBox.DrawingArea` and :class:`~OffsetBox.TextArea`.
The `TextArea` calculates the required space for the text and so the
text is always within the bounds, for this nothing has changed.

However, `DrawingArea` acts as a parent for zero or more `Artists` that
draw on it and may do so beyond the bounds. Now child `Artists` can be
clipped to the bounds of the `DrawingArea`.
