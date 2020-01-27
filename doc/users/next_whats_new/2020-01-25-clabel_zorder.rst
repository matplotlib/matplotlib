Set zorder of contour labels
----------------------------
`~.axes.Axes.clabel` now accepts a ``zorder`` kwarg
making it easier to set the ``zorder`` of contour labels.
If not specified, the default ``zorder`` of clabels used to always be 3
(i.e. the default ``zorder`` of `~.text.Text`) irrespective of the ``zorder``
passed to `~.axes.Axes.contour`/`~.axes.Axes.contourf`.
The new default ``zorder`` for clabels has been changed to (2 + ``zorder``
passed to `~.axes.Axes.contour`/`~.axes.Axes.contourf`).
