

API Changes in 2.1.0
====================


Default behavior of log scales changed to mask <= 0 values
----------------------------------------------------------

Calling `matplotlib.axes.Axes.set_xscale` or `matplotlib.axes.Axes.set_yscale`
now uses 'mask' as the default method to handle invalid values (as opposed to
'clip'). This means that any values <= 0 on a log scale will not be shown.

Previously they were clipped to a very small number and shown.


:meth:`matplotlib.cbook.CallbackRegistry.process` suppresses exceptions by default
----------------------------------------------------------------------------------

Matplotlib uses instances of :obj:`~matplotlib.cbook.CallbackRegistry`
as a bridge between user input event from the GUI and user callbacks.
Previously, any exceptions raised in a user call back would bubble out
of the ``process`` method, which is typically in the GUI event
loop.  Most GUI frameworks simple print the traceback to the screen
and continue as there is not always a clear method of getting the
exception back to the user.  However PyQt5 now exits the process when
it receives an un-handled python exception in the event loop.  Thus,
:meth:`~matplotlib.cbook.CallbackRegistry.process` now suppresses and
prints tracebacks to stderr by default.

What :meth:`~matplotlib.cbook.CallbackRegistry.process` does with exceptions
is now user configurable via the ``exception_handler`` attribute and kwarg.  To
restore the previous behavior pass ``None`` ::

  cb = CallbackRegistry(exception_handler=None)


A function which take and ``Exception`` as its only argument may also be passed ::

  def maybe_reraise(exc):
      if isinstance(exc, RuntimeError):
          pass
      else:
          raise exc

  cb = CallbackRegistry(exception_handler=maybe_reraise)



Improved toggling of the axes grids
-----------------------------------

The ``g`` key binding now switches the states of the ``x`` and ``y`` grids
independently (by cycling through all four on/off combinations).

The new ``G`` key binding switches the states of the minor grids.

Both bindings are disabled if only a subset of the grid lines (in either
direction) is visible, to avoid making irreversible changes to the figure.


Ticklabels are turned off instead of being invisible
----------------------------------------------------

Internally, the `.Tick`'s ``~matplotlib.axis.Tick.label1On`` attribute
is now used to hide tick labels instead of setting the visibility on the tick
label objects.
This improves overall performance and fixes some issues.
As a consequence, in case those labels ought to be shown,
:func:`~matplotlib.axes.Axes.tick_params`
needs to be used, e.g.

::

    ax.tick_params(labelbottom=True)


Removal of warning on empty legends
-----------------------------------

`.pyplot.legend` used to issue a warning when no labeled artist could be
found.  This warning has been removed.


More accurate legend autopositioning
------------------------------------

Automatic positioning of legends now prefers using the area surrounded
by a `.Line2D` rather than placing the legend over the line itself.


Cleanup of stock sample data
----------------------------

The sample data of stocks has been cleaned up to remove redundancies and
increase portability. The ``AAPL.dat.gz``, ``INTC.dat.gz`` and ``aapl.csv``
files have been removed entirely and will also no longer be available from
`matplotlib.cbook.get_sample_data`. If a CSV file is required, we suggest using
the ``msft.csv`` that continues to be shipped in the sample data. If a NumPy
binary file is acceptable, we suggest using one of the following two new files.
The ``aapl.npy.gz`` and ``goog.npy`` files have been replaced by ``aapl.npz``
and ``goog.npz``, wherein the first column's type has changed from
`datetime.date` to `numpy.datetime64` for better portability across Python
versions. Note that Matplotlib does not fully support `numpy.datetime64` as
yet.


Updated qhull to 2015.2
-----------------------

The version of qhull shipped with Matplotlib, which is used for
Delaunay triangulation, has been updated from version 2012.1 to
2015.2.

Improved Delaunay triangulations with large offsets
---------------------------------------------------

Delaunay triangulations now deal with large x,y offsets in a better
way. This can cause minor changes to any triangulations calculated
using Matplotlib, i.e. any use of `matplotlib.tri.Triangulation` that
requests that a Delaunay triangulation is calculated, which includes
`matplotlib.pyplot.tricontour`, `matplotlib.pyplot.tricontourf`,
`matplotlib.pyplot.tripcolor`, `matplotlib.pyplot.triplot`,
``matplotlib.mlab.griddata`` and
`mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`.



Use ``backports.functools_lru_cache`` instead of ``functools32``
----------------------------------------------------------------

It's better maintained and more widely used (by pylint, jaraco, etc).



``cbook.is_numlike`` only performs an instance check
----------------------------------------------------

``matplotlib.cbook.is_numlike`` now only checks that its argument
is an instance of ``(numbers.Number, np.Number)``.  In particular,
this means that arrays are now not num-like.



Elliptical arcs now drawn between correct angles
------------------------------------------------

The `matplotlib.patches.Arc` patch is now correctly drawn between the given
angles.

Previously a circular arc was drawn and then stretched into an ellipse,
so the resulting arc did not lie between *theta1* and *theta2*.



``-d$backend`` no longer sets the backend
-----------------------------------------

It is no longer possible to set the backend by passing ``-d$backend``
at the command line.  Use the ``MPLBACKEND`` environment variable
instead.


Path.intersects_bbox always treats the bounding box as filled
-------------------------------------------------------------

Previously, when ``Path.intersects_bbox`` was called with ``filled`` set to
``False``, it would treat both the path and the bounding box as unfilled. This
behavior was not well documented and it is usually not the desired behavior,
since bounding boxes are used to represent more complex shapes located inside
the bounding box. This behavior has now been changed: when ``filled`` is
``False``, the path will be treated as unfilled, but the bounding box is still
treated as filled. The old behavior was arguably an implementation bug.

When ``Path.intersects_bbox`` is called with ``filled`` set to ``True``
(the default value), there is no change in behavior. For those rare cases where
``Path.intersects_bbox`` was called with ``filled`` set to ``False`` and where
the old behavior is actually desired, the suggested workaround is to call
``Path.intersects_path`` with a rectangle as the path::

    from matplotlib.path import Path
    from matplotlib.transforms import Bbox, BboxTransformTo
    rect = Path.unit_rectangle().transformed(BboxTransformTo(bbox))
    result = path.intersects_path(rect, filled=False)




WX no longer calls generates ``IdleEvent`` events or calls ``idle_event``
-------------------------------------------------------------------------

Removed unused private method ``_onIdle`` from ``FigureCanvasWx``.

The ``IdleEvent`` class and ``FigureCanvasBase.idle_event`` method
will be removed in 2.2



Correct scaling of ``magnitude_spectrum()``
-------------------------------------------

The functions :func:`matplotlib.mlab.magnitude_spectrum()` and :func:`matplotlib.pyplot.magnitude_spectrum()` implicitly assumed the sum
of windowing function values to be one. In Matplotlib and Numpy the
standard windowing functions are scaled to have maximum value of one,
which usually results in a sum of the order of n/2 for a n-point
signal. Thus the amplitude scaling ``magnitude_spectrum()`` was
off by that amount when using standard windowing functions (`Bug 8417
<https://github.com/matplotlib/matplotlib/issues/8417>`_ ). Now the
behavior is consistent with :func:`matplotlib.pyplot.psd()` and
:func:`scipy.signal.welch()`. The following example demonstrates the
new and old scaling::

    import matplotlib.pyplot as plt
    import numpy as np

    tau, n = 10, 1024  # 10 second signal with 1024 points
    T = tau/n  # sampling interval
    t = np.arange(n)*T

    a = 4  # amplitude
    x = a*np.sin(40*np.pi*t)  # 20 Hz sine with amplitude a

    # New correct behavior: Amplitude at 20 Hz is a/2
    plt.magnitude_spectrum(x, Fs=1/T, sides='onesided', scale='linear')

    # Original behavior: Amplitude at 20 Hz is (a/2)*(n/2) for a Hanning window
    w = np.hanning(n)  # default window is a Hanning window
    plt.magnitude_spectrum(x*np.sum(w), Fs=1/T, sides='onesided', scale='linear')





Change to signatures of :meth:`~matplotlib.axes.Axes.bar` & :meth:`~matplotlib.axes.Axes.barh`
----------------------------------------------------------------------------------------------

For 2.0 the :ref:`default value of *align* <barbarh_align>` changed to
``'center'``.  However this caused the signature of
:meth:`~matplotlib.axes.Axes.bar` and
:meth:`~matplotlib.axes.Axes.barh` to be misleading as the first parameters were
still *left* and *bottom* respectively::

  bar(left, height, *, align='center', **kwargs)
  barh(bottom, width, *, align='center', **kwargs)

despite behaving as the center in both cases. The methods now take
``*args, **kwargs`` as input and are documented to have the primary
signatures of::

  bar(x, height, *, align='center', **kwargs)
  barh(y, width, *, align='center', **kwargs)

Passing *left* and *bottom* as keyword arguments to
:meth:`~matplotlib.axes.Axes.bar` and
:meth:`~matplotlib.axes.Axes.barh` respectively will warn.
Support will be removed in Matplotlib 3.0.


Font cache as json
------------------

The font cache is now saved as json, rather than a pickle.


Invalid (Non-finite) Axis Limit Error
-------------------------------------

When using :func:`~matplotlib.axes.Axes.set_xlim` and
:func:`~matplotlib.axes.Axes.set_ylim`, passing non-finite values now
results in a ``ValueError``. The previous behavior resulted in the
limits being erroneously reset to ``(-0.001, 0.001)``.

``scatter`` and ``Collection`` offsets are no longer implicitly flattened
-------------------------------------------------------------------------

`~matplotlib.collections.Collection` (and thus both 2D
`~matplotlib.axes.Axes.scatter` and 3D
`~mpl_toolkits.mplot3d.axes3d.Axes3D.scatter`) no
longer implicitly flattens its offsets.  As a consequence, ``scatter``'s ``x``
and ``y`` arguments can no longer be 2+-dimensional arrays.

Deprecations
------------

``GraphicsContextBase``\'s ``linestyle`` property.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GraphicsContextBase.get_linestyle`` and
``GraphicsContextBase.set_linestyle`` methods, which had no effect,
have been deprecated.  All of the backends Matplotlib ships use
``GraphicsContextBase.get_dashes`` and
``GraphicsContextBase.set_dashes`` which are more general.
Third-party backends should also migrate to the ``*_dashes`` methods.


``NavigationToolbar2.dynamic_update``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use `~.FigureCanvasBase.draw_idle` method on the ``Canvas`` instance instead.


Testing
~~~~~~~

``matplotlib.testing.noseclasses`` is deprecated and will be removed in 2.3


``EngFormatter`` *num* arg as string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing a string as *num* argument when calling an instance of
`matplotlib.ticker.EngFormatter` is deprecated and will be removed in 2.3.


``mpl_toolkits.axes_grid`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All functionally from ``mpl_toolkits.axes_grid`` can be found in either
`mpl_toolkits.axes_grid1` or `mpl_toolkits.axisartist`. Axes classes
from ``mpl_toolkits.axes_grid`` based on ``Axis`` from
`mpl_toolkits.axisartist` can be found in `mpl_toolkits.axisartist`.


``Axes`` collision in ``Figure.add_axes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding an axes instance to a figure by using the same arguments as for
a previous axes instance currently reuses the earlier instance.  This
behavior has been deprecated in Matplotlib 2.1. In a future version, a
*new* instance will always be created and returned.  Meanwhile, in such
a situation, a deprecation warning is raised by
``matplotlib.figure.AxesStack``.

This warning can be suppressed, and the future behavior ensured, by passing
a *unique* label to each axes instance.  See the docstring of
:meth:`~matplotlib.figure.Figure.add_axes` for more information.

Additional details on the rationale behind this deprecation can be found
in :ghissue:`7377` and :ghissue:`9024`.


Former validators for ``contour.negative_linestyle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The former public validation functions ``validate_negative_linestyle``
and ``validate_negative_linestyle_legacy`` will be deprecated in 2.1 and
may be removed in 2.3. There are no public functions to replace them.



``cbook``
~~~~~~~~~

Many unused or near-unused :mod:`matplotlib.cbook` functions and
classes have been deprecated: ``converter``, ``tostr``,
``todatetime``, ``todate``, ``tofloat``, ``toint``, ``unique``,
``is_string_like``, ``is_sequence_of_strings``, ``is_scalar``,
``Sorter``, ``Xlator``, ``soundex``, ``Null``, ``dict_delall``,
``RingBuffer``, ``get_split_ind``, ``wrap``,
``get_recursive_filelist``, ``pieces``, ``exception_to_str``,
``allequal``, ``alltrue``, ``onetrue``, ``allpairs``, ``finddir``,
``reverse_dict``, ``restrict_dict``, ``issubclass_safe``,
``recursive_remove``, ``unmasked_index_ranges``.


Code Removal
------------

qt4_compat.py
~~~~~~~~~~~~~

Moved to ``qt_compat.py``.  Renamed because it now handles Qt5 as well.


Previously Deprecated methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GraphicsContextBase.set_graylevel``, ``FigureCanvasBase.onHilite`` and
``mpl_toolkits.axes_grid1.mpl_axes.Axes.toggle_axisline`` methods have been
removed.

The ``ArtistInspector.findobj`` method, which was never working due to the lack
of a ``get_children`` method, has been removed.

The deprecated ``point_in_path``, ``get_path_extents``,
``point_in_path_collection``, ``path_intersects_path``,
``convert_path_to_polygons``, ``cleanup_path`` and ``clip_path_to_rect``
functions in the ``matplotlib.path`` module have been removed.  Their
functionality remains exposed as methods on the ``Path`` class.

The deprecated ``Artist.get_axes`` and ``Artist.set_axes`` methods
have been removed


The ``matplotlib.backends.backend_ps.seq_allequal`` function has been removed.
Use ``np.array_equal`` instead.

The deprecated ``matplotlib.rcsetup.validate_maskedarray``,
``matplotlib.rcsetup.deprecate_savefig_extension`` and
``matplotlib.rcsetup.validate_tkpythoninspect`` functions, and associated
``savefig.extension`` and ``tk.pythoninspect`` rcparams entries have been
removed.


The keyword argument *resolution* of
:class:`matplotlib.projections.polar.PolarAxes` has been removed. It
has deprecation with no effect from version *0.98.x*.


``Axes.set_aspect("normal")``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for setting an ``Axes``\'s aspect to ``"normal"`` has been
removed, in favor of the synonym ``"auto"``.


``shading`` kwarg to ``pcolor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``shading`` kwarg to `~matplotlib.axes.Axes.pcolor` has been
removed.  Set ``edgecolors`` appropriately instead.


Functions removed from the ``lines`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`matplotlib.lines` module no longer imports the
``pts_to_prestep``, ``pts_to_midstep`` and ``pts_to_poststep``
functions from :mod:`matplotlib.cbook`.


PDF backend functions
~~~~~~~~~~~~~~~~~~~~~

The methods ``embedTeXFont`` and ``tex_font_mapping`` of
:class:`matplotlib.backends.backend_pdf.PdfFile` have been removed.  It is
unlikely that external users would have called these methods, which
are related to the font system internal to the PDF backend.


matplotlib.delaunay
~~~~~~~~~~~~~~~~~~~

Remove the delaunay triangulation code which is now handled by Qhull
via :mod:`matplotlib.tri`.
