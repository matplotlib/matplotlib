'OffsetBox.DrawingArea' respects the 'clip' keyword argument
````````````````````````````````````````````````````````````

The call signature was `OffsetBox.DrawingArea(..., clip=True)` but nothing
was done with the `clip` argument. The object did not do any clipping
regardless of that parameter. Now the object can and does clip the child `Artists` if they are set to be clipped.

You can turn off the clipping on a per-child basis using `child.set_clip_on(False)`.
