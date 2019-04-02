PathCollections created with `~.Axes.scatter` now keep track of invalid points
``````````````````````````````````````````````````````````````````````````````

Previously, points with nonfinite (infinite or nan) coordinates would not be
included in the offsets (as returned by `PathCollection.get_offsets`) of a
`PathCollection` created by `~.Axes.scatter`, and points with nonfinite values
(as specified by the *c* kwarg) would not be included in the array (as returned
by `PathCollection.get_array`)

Such points are now included, but masked out by returning a masked array.

If the *plotnonfinite* kwarg to `~.Axes.scatter` is set, then points with
nonfinite values are plotted using the bad color of the `PathCollection`\ 's
colormap (as set by `Colormap.set_bad`).
