`rcount` and `ccount` for `plot_surface()`
------------------------------------------

As of v2.0, mplot3d's :func:`~mpl_toolkits.mplot3d.axes3d.plot_surface` now
accepts `rcount` and `ccount` arguments for controlling the sampling of the
input data for plotting. These arguments specify the maximum number of
evenly spaced samples to take from the input data. These arguments are
also the new default sampling method for the function, and is
considered a style change.

The old `rstride` and `cstride` arguments, which specified the size of the
evenly spaced samples, become the default when 'classic' mode is invoked,
and are still available for use. There are no plans for deprecating these
arguments.

