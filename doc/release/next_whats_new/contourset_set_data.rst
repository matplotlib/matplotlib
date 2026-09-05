Contour sets can be updated with new data
-----------------------------------------
`.ContourSet.set_data` recontours new data using the existing artist, instead of
requiring the contour set to be removed and recreated::

    cs = ax.contour(X, Y, Z)
    cs.set_data(X, Y, Z2)

It works for `~.Axes.contour`, `~.Axes.contourf`, `~.Axes.tricontour` and
`~.Axes.tricontourf`. The contour levels are kept, so the colors, the colorbar
and the legend entries of the contour set remain valid. This is useful for
animations and other interactive updates.
