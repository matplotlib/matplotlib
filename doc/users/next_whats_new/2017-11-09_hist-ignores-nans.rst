Histogram function now accepts nan values in input
--------------------------------------------------

The `~.Axes.hist` function now accepts nan values in both the *data* and
*weights* input. Previously this would just error. Now any invalid values
are simply ignored when calculating the histogram values.
