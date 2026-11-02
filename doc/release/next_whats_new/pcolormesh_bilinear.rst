``pcolormesh`` supports bilinear shading
----------------------------------------

`~.Axes.pcolormesh` now accepts ``shading='bilinear'`` to interpolate the
mapped RGBA colors over each quadrilateral using all four corner colors.
Unlike Gouraud shading, this interpolation does not split each quadrilateral
into triangles, avoiding triangulation-dependent patterns.

The data and coordinate grids must have the same shape, with at least two rows
and two columns. Each cell must be a nondegenerate parallelogram (including
rectangles), and the mesh transform must be affine. Unsupported geometry or
non-affine transforms raise a `ValueError`. Cells with a masked corner are
transparent, and edge colors are ignored. Vector output formats render bilinear
shading as an image at the DPI requested when saving the figure.

For example::

    import matplotlib.pyplot as plt
    import numpy as np

    X, Y = np.meshgrid(np.arange(6), np.arange(6))
    Z = X + 2*Y
    Z[2, 3] = 15
    plt.pcolormesh(X, Y, Z, shading='bilinear')
    plt.show()
