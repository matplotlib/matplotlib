`Axes.imshow` clips RGB values to the valid range
-------------------------------------------------

When `Axes.imshow` is passed an RGB or RGBA value with out-of-range
values, it now logs a warning and clips them to the valid range.
The old behaviour, wrapping back in to the range, often hid outliers
and made interpreting RGB images unreliable.
