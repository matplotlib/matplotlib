Short lines remain visible with automatic pixel snapping
--------------------------------------------------------

Short horizontal and vertical lines no longer disappear from raster output
because their endpoints snap to the same pixel. Explicitly setting ``snap=True``
continues to force pixel snapping.
