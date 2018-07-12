Changes for 0.84
================

.. code-block:: text

    Unified argument handling between hlines and vlines.  Both now
    take optionally a fmt argument (as in plot) and a keyword args
    that can be passed onto Line2D.

    Removed all references to "data clipping" in rc and lines.py since
    these were not used and not optimized.  I'm sure they'll be
    resurrected later with a better implementation when needed.

    'set' removed - no more deprecation warnings.  Use 'setp' instead.

    Backend developers: Added flipud method to image and removed it
    from to_str.  Removed origin kwarg from backend.draw_image.
    origin is handled entirely by the frontend now.
