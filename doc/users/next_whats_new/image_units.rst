Unit support for images
-----------------------
This adds support for image data with units has been added to the following plotting
methods:

- `~.axes.Axes.imshow`
- `~.axes.Axes.pcolor`
- `~.axes.Axes.pcolormesh`

If the data has units, the ``vmin`` and ``vmax`` units to these methods can also have
units, and if you add a colorbar the ``levels`` argument to ``colorbar`` can also
have units.
