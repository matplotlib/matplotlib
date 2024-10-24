Passing floating-point values to ``RendererAgg.draw_text_image``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any floating-point values passed to the *x* and *y* parameters were truncated to integers
silently. This behaviour is now deprecated, and only `int` values should be used.

Passing floating-point values to ``FT2Image``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any floating-point values passed to the `.FT2Image` constructor, or the *x0*, *y0*, *x1*,
and *y1* parameters of `.FT2Image.draw_rect_filled` were truncated to integers silently.
This behaviour is now deprecated, and only `int` values should be used.
