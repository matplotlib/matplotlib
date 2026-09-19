``Axes.hist2d`` now converts *x*/*y*/*range* through the axis unit converters
------------------------------------------------------------------------------

`~matplotlib.axes.Axes.hist2d` previously passed *x* and *y* directly to
`numpy.histogram2d` without applying the axes' unit converters, unlike
`~.Axes.plot`, `~.Axes.scatter`, and `~.Axes.hist`. This meant that, e.g.,
plotting ``datetime64`` data with `~.Axes.hist2d` produced bin edges in raw
(often nanosecond-scale) units instead of Matplotlib's internal date
representation, so the histogram did not line up with other artists plotted
on the same Axes. ``x``, ``y``, and ``range`` (if passed) are now converted
consistently with the other plotting methods.
