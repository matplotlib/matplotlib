Add :rc:`contour.linewidth` to rcParams
---------------------------------------

The new config option :rc:`contour.linewidth` allows to control the default
linewidth of contours as a float. When set to ``None``, the linewidths fall
back to :rc:`lines.linewidth`. The config value is overidden as usual
by the ``linewidths`` argument passed to `~.axes.Axes.contour` when
it is not set to ``None``.

