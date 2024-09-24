Passing floating-point values to ``RendererAgg.draw_text_image``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any floating-point values passed to the *x* and *y* parameters were truncated to integers
silently. This behaviour is now deprecated, and only `int` values should be used.
