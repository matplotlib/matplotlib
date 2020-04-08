``Axes3D`` no longer distorts the 3d plot to match the 2d aspect ratio
----------------------------------------------------------------------

Plots made with :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` were previously
stretched to fit a square bounding box. As this stretching was done after
the projection from 3D to 2D, it resulted in distorted images if non-square
bounding boxes were used.

As of this release, this no longer occurs.
