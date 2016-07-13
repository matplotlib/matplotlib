New rcParams
------------

The parameters ``xtick.top``, ``xtick.bottom``, ``ytick.left``
and ``ytick.right`` were added to control where the ticks
are drawn.

Furthermore, the parameters ``xtick.major.top``, ``xtick.major.bottom``,
``xtick.minor.top``, ``xtick.minor.bottom``, ``ytick.major.left``,
``ytick.major.right``, ``ytick.minor.left``, and ``ytick.minor.right`` were
added to control were ticks are drawn.

``hist.bins`` to control the default number of bins to use in
`~matplotlib.axes.Axes.hist`.  This can be an `int`, a list of floats, or
``'auto'`` if numpy >= 1.11 is installed.
