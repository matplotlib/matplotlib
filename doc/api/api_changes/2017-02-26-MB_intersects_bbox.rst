Path.intersects_bbox always treats the bounding box as filled
`````````````````````````````````````````````````````````````

Previously, when ``Path.intersects_bbox`` was called with ``filled`` set to
``False``, it would treat both the path and the bounding box as unfilled. This
behavior was not well documented and it is usually not the desired behavior,
since bounding boxes are used to represent more complex shapes located inside
the bounding box. This behavior has now been changed: when ``filled`` is
``False``, the path will be treated as unfilled, but the bounding box is still
treated as filled. The old behavior was arguably an implementation bug.

When ``Path.intersects_bbox`` is called with ``filled`` set to ``True``
(the default value), there is no change in behavior. For those rare cases where
``Path.intersects_bbox`` was called with ``filled`` set to ``False`` and where
the old behavior is actually desired, the suggested workaround is to call
``Path.intersects_path`` with a rectangle as the path::

    from matplotlib.path import Path
    from matplotlib.transforms import Bbox, BboxTransformTo
    rect = Path.unit_rectangle().transformed(BboxTransformTo(bbox))
    result = path.intersects_path(rect, filled=False)
