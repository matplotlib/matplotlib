Pcolor and Pcolormesh now accept shading='nearest' and 'auto'
-------------------------------------------------------------

Previously `.axes.Axes.pcolor` and  `.axes.Axes.pcolormesh` handled
the situation where *x* and *y* have the same (respective) size as *C* by
dropping the last row and column of *C*, and *x* and *y* are regarded as the
edges of the remaining rows and columns in *C*.  However, many users want
*x* and *y* centered on the rows and columns of *C*.

To accommodate this, ``shading='nearest'`` and ``shading='auto'`` are
new allowed strings for the ``shading`` kwarg.  ``'nearest'`` will center the
color on *x* and *y* if *x* and *y* have the same dimensions as *C*
(otherwise an error will be thrown).  ``shading='auto'`` will choose 'flat'
or 'nearest' based on the size of *X*, *Y*, *C*.

If ``shading='flat'`` then *X*, and *Y* should have dimensions one larger
than *C*.  If *X* and *Y* have the same dimensions as *C*, then the previous
behavior is used and the last row and column of *C* are dropped, and a
DeprecationWarning is emitted.

Users can also specify this by the new :rc:`pcolor.shading` in their
``.matplotlibrc`` or via `.rcParams`.

See :doc:`pcolormesh </gallery/images_contours_and_fields/pcolormesh_grids>`
for examples.
