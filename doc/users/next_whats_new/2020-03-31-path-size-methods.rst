
Functions to compute a Path's size
----------------------------------

Various functions were added to `~.bezier.BezierSegment` and `~.path.Path` to
allow computation of the shape, size and area of a `~.path.Path` and its
composite Bezier curves.

In addition to the fixes below, `~.bezier.BezierSegment` has gained more
documentation and usability improvements, including properties that contain its
dimension, degree, control_points, and more.

Better interface for Path segment iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~.path.Path.iter_bezier` iterates through the `~.bezier.BezierSegment`'s that
make up the Path. This is much more useful typically than the existing
`~.path.Path.iter_segments` function, which returns the absolute minimum amount
of information possible to reconstruct the Path.

Fixed bug that computed a Path's Bbox incorrectly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Historically, `~.path.Path.get_extents` has always simply returned the Bbox of
a curve's control points, instead of the Bbox of the curve itself. While this is
a correct upper bound for the path's extents, it can differ dramatically from
the Path's actual extents for non-linear Bezier curves.

Path area
~~~~~~~~~

A `~.path.Path.signed_area` method was added to compute the signed filled area
of a Path object analytically (i.e. without integration). This should be useful
for constructing Paths of a desired area.
