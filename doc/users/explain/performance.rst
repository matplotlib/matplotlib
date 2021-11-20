.. _performance:

Performance
===========

Whether exploring data in interactive mode or programmatically
saving lots of plots, rendering performance can be a challenging
bottleneck in your pipeline. Matplotlib provides multiple
ways to greatly reduce rendering time at the cost of a slight
change (to a settable tolerance) in your plot's appearance.
The methods available to reduce rendering time depend on the
type of plot that is being created.

Line segment simplification
---------------------------

For plots that have line segments (e.g. typical line plots, outlines
of polygons, etc.), rendering performance can be controlled by
:rc:`path.simplify` and :rc:`path.simplify_threshold`, which
can be defined e.g. in the :file:`matplotlibrc` file (see
:doc:`/tutorials/introductory/customizing` for more information about
the :file:`matplotlibrc` file). :rc:`path.simplify` is a Boolean
indicating whether or not line segments are simplified at all.
:rc:`path.simplify_threshold` controls how much line segments are simplified;
higher thresholds result in quicker rendering.

The following script will first display the data without any
simplification, and then display the same data with simplification.
Try interacting with both of them::

  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib as mpl

  # Setup, and create the data to plot
  y = np.random.rand(100000)
  y[50000:] *= 2
  y[np.geomspace(10, 50000, 400).astype(int)] = -1
  mpl.rcParams['path.simplify'] = True

  mpl.rcParams['path.simplify_threshold'] = 0.0
  plt.plot(y)
  plt.show()

  mpl.rcParams['path.simplify_threshold'] = 1.0
  plt.plot(y)
  plt.show()

Matplotlib currently defaults to a conservative simplification
threshold of ``1/9``. To change default settings to use a different
value, change the :file:`matplotlibrc` file. Alternatively, users
can create a new style for interactive plotting (with maximal
simplification) and another style for publication quality plotting
(with minimal simplification) and activate them as necessary. See
:doc:`/tutorials/introductory/customizing` for instructions on
how to perform these actions.


The simplification works by iteratively merging line segments
into a single vector until the next line segment's perpendicular
distance to the vector (measured in display-coordinate space)
is greater than the ``path.simplify_threshold`` parameter.

.. note::
  Changes related to how line segments are simplified were made
  in version 2.1. Rendering time will still be improved by these
  parameters prior to 2.1, but rendering time for some kinds of
  data will be vastly improved in versions 2.1 and greater.

Marker simplification
---------------------

Markers can also be simplified, albeit less robustly than
line segments. Marker simplification is only available
to :class:`~matplotlib.lines.Line2D` objects (through the
``markevery`` property). Wherever
:class:`~matplotlib.lines.Line2D` construction parameters
are passed through, such as
:func:`matplotlib.pyplot.plot` and
:meth:`matplotlib.axes.Axes.plot`, the ``markevery``
parameter can be used::

  plt.plot(x, y, markevery=10)

The ``markevery`` argument allows for naive subsampling, or an
attempt at evenly spaced (along the *x* axis) sampling. See the
:doc:`/gallery/lines_bars_and_markers/markevery_demo`
for more information.

Splitting lines into smaller chunks
-----------------------------------

If you are using the Agg backend (see :ref:`what-is-a-backend`),
then you can make use of :rc:`agg.path.chunksize`
This allows users to specify a chunk size, and any lines with
greater than that many vertices will be split into multiple
lines, each of which has no more than ``agg.path.chunksize``
many vertices. (Unless ``agg.path.chunksize`` is zero, in
which case there is no chunking.) For some kind of data,
chunking the line up into reasonable sizes can greatly
decrease rendering time.

The following script will first display the data without any
chunk size restriction, and then display the same data with
a chunk size of 10,000. The difference can best be seen when
the figures are large, try maximizing the GUI and then
interacting with them::

  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  mpl.rcParams['path.simplify_threshold'] = 1.0

  # Setup, and create the data to plot
  y = np.random.rand(100000)
  y[50000:] *= 2
  y[np.geomspace(10, 50000, 400).astype(int)] = -1
  mpl.rcParams['path.simplify'] = True

  mpl.rcParams['agg.path.chunksize'] = 0
  plt.plot(y)
  plt.show()

  mpl.rcParams['agg.path.chunksize'] = 10000
  plt.plot(y)
  plt.show()

Legends
-------

The default legend behavior for axes attempts to find the location
that covers the fewest data points (``loc='best'``). This can be a
very expensive computation if there are lots of data points. In
this case, you may want to provide a specific location.

Using the *fast* style
----------------------

The *fast* style can be used to automatically set
simplification and chunking parameters to reasonable
settings to speed up plotting large amounts of data.
The following code runs it::

  import matplotlib.style as mplstyle
  mplstyle.use('fast')

It is very lightweight, so it works well with other
styles. Be sure the fast style is applied last
so that other styles do not overwrite the settings::

  mplstyle.use(['dark_background', 'ggplot', 'fast'])
