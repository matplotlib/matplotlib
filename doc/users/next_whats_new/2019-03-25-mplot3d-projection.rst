``Axes3D`` no longer distorts the 3d plot to match the 2d aspect ratio
----------------------------------------------------------------------

Plots made with :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` were previously
stretched to fit a square bounding box. As this stretching was done after
the projection from 3D to 2D, it resulted in distorted images if non-square
bounding boxes were used.  As of 3.3, this no longer occurs.

Currently, modes of setting the aspect (via
`~mpl_toolkits.mplot3d.axes3d.Axes3D.set_aspect`) in data space are
not supported for Axes3D but may be in the future.  If you want to
simulate having equal aspect in data space, set the ratio of your data
limits to match the value of `~.get_box_aspect`.  To control these
ratios use the `~mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect`
method which accepts the ratios as a 3-tuple of X:Y:Z.  The default
aspect ratio is 4:4:3.
