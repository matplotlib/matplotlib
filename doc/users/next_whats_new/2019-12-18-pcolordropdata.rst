Pcolor and Pcolormesh now have *dropdata* kwarg and rcParam
-----------------------------------------------------------

Previously `.axes.Axes.pcolor` and  `.axes.Axes.pcolormesh` handled
the situation where *x* and *y* have the same (respective) size as *Z* by
dropping the last row and column of *Z*, and *x* and *y* are regarded as the
edges of the remaining rows and columns in *Z*.  However, most users probably
really want *x* and *y* centered on the rows and columns of *Z*, so if
they specify *dropdata* as True, both methods will now linearly interpolate to
get the edges of the bins, and *x* and *y* will specify the (linear) center of
each gridcell in the pseudocolor plot.

Users can also specify this by the new :rc:`pcolor.dropdata` in their
``.matplotlibrc`` or via `.rcParams`.

See :doc:`pcolormesh </gallery/images_contours_and_fields/pcolormesh_levels>`
for an example.  
