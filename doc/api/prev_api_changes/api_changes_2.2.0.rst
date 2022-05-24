
API Changes in 2.2.0
====================



New dependency
--------------

`kiwisolver <https://github.com/nucleic/kiwi>`__ is now a required
dependency to support the new constrained_layout,  see
:doc:`/tutorials/intermediate/constrainedlayout_guide` for
more details.


Deprecations
------------

Classes, functions, and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unused and untested ``Artist.onRemove`` and ``Artist.hitlist`` methods have
been deprecated.

The now unused ``mlab.less_simple_linear_interpolation`` function is
deprecated.

The unused ``ContourLabeler.get_real_label_width`` method is deprecated.

The unused ``FigureManagerBase.show_popup`` method is deprecated.  This
introduced in e945059b327d42a99938b939a1be867fa023e7ba in 2005 but never built
out into any of the backends.

``backend_tkagg.AxisMenu`` is deprecated, as it has become unused since the
removal of "classic" toolbars.


Changed function signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

kwarg ``fig`` to `.GridSpec.get_subplot_params` is
deprecated,  use ``figure`` instead.

Using `.pyplot.axes` with an `~matplotlib.axes.Axes` as argument is deprecated. This sets
the current axes, i.e. it has the same effect as `.pyplot.sca`. For clarity
``plt.sca(ax)`` should be preferred over ``plt.axes(ax)``.


Using strings instead of booleans to control grid and tick visibility
is deprecated.  Using ``"on"``, ``"off"``, ``"true"``, or ``"false"``
to control grid and tick visibility has been deprecated.  Instead, use
normal booleans (``True``/``False``) or boolean-likes.  In the future,
all non-empty strings may be interpreted as ``True``.

When given 2D inputs with non-matching numbers of columns, `~.pyplot.plot`
currently cycles through the columns of the narrower input, until all the
columns of the wider input have been plotted.  This behavior is deprecated; in
the future, only broadcasting (1 column to *n* columns) will be performed.


rcparams
~~~~~~~~

The :rc:`backend.qt4` and :rc:`backend.qt5` rcParams were deprecated
in version 2.2.  In order to force the use of a specific Qt binding,
either import that binding first, or set the ``QT_API`` environment
variable.

Deprecation of the ``nbagg.transparent`` rcParam.  To control
transparency of figure patches in the nbagg (or any other) backend,
directly set ``figure.patch.facecolor``, or the ``figure.facecolor``
rcParam.

Deprecated ``Axis.unit_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``Axis.units`` (which has long existed) instead.


Removals
--------

Function Signatures
~~~~~~~~~~~~~~~~~~~

Contouring no longer supports ``legacy`` corner masking.  The
deprecated ``ContourSet.vmin`` and ``ContourSet.vmax`` properties have
been removed.

Passing ``None`` instead of ``"none"`` as format to `~.Axes.errorbar` is no
longer supported.

The ``bgcolor`` keyword argument to ``Axes`` has been removed.

Modules, methods, and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``matplotlib.finance``, ``mpl_toolkits.exceltools`` and
``mpl_toolkits.gtktools`` modules have been removed.  ``matplotlib.finance``
remains available at https://github.com/matplotlib/mpl_finance.

The ``mpl_toolkits.mplot3d.art3d.iscolor`` function has been removed.

The ``Axes.get_axis_bgcolor``, ``Axes.set_axis_bgcolor``,
``Bbox.update_from_data``, ``Bbox.update_datalim_numerix``,
``MaxNLocator.bin_boundaries`` methods have been removed.

``mencoder`` can no longer be used to encode animations.

The unused ``FONT_SCALE`` and ``fontd`` attributes of the `.RendererSVG`
class have been removed.

colormaps
~~~~~~~~~

The ``spectral`` colormap has been removed.  The ``Vega*`` colormaps, which
were aliases for the ``tab*`` colormaps, have been removed.


rcparams
~~~~~~~~

The following deprecated rcParams have been removed:

- ``axes.color_cycle`` (see ``axes.prop_cycle``),
- ``legend.isaxes``,
- ``svg.embed_char_paths`` (see ``svg.fonttype``),
- ``text.fontstyle``, ``text.fontangle``, ``text.fontvariant``,
  ``text.fontweight``, ``text.fontsize`` (renamed to ``text.style``, etc.),
- ``tick.size`` (renamed to ``tick.major.size``).



Only accept string-like for Categorical input
---------------------------------------------

Do not accept mixed string / float / int input, only
strings are valid categoricals.

Removal of unused imports
-------------------------
Many unused imports were removed from the codebase.  As a result,
trying to import certain classes or functions from the "wrong" module
(e.g. `~.Figure` from :mod:`matplotlib.backends.backend_agg` instead of
:mod:`matplotlib.figure`) will now raise an `ImportError`.


``Axes3D.get_xlim``, ``get_ylim`` and ``get_zlim`` now return a tuple
---------------------------------------------------------------------

They previously returned an array.  Returning a tuple is consistent with the
behavior for 2D axes.


Exception type changes
----------------------

If `.MovieWriterRegistry` can't find the requested `.MovieWriter`, a
more helpful `RuntimeError` message is now raised instead of the
previously raised `KeyError`.

``matplotlib.tight_layout.auto_adjust_subplotpars`` now raises `ValueError`
instead of `RuntimeError` when sizes of input lists don't match


`.Figure.set_figwidth` and `.Figure.set_figheight` default *forward* to True
----------------------------------------------------------------------------

`matplotlib.figure.Figure.set_figwidth` and
`matplotlib.figure.Figure.set_figheight` had the keyword argument
``forward=False`` by default, but `.figure.Figure.set_size_inches` now defaults
to ``forward=True``.  This makes these functions consistent.


Do not truncate svg sizes to nearest point
------------------------------------------

There is no reason to size the SVG out put in integer points, change
to out putting floats for the *height*, *width*, and *viewBox* attributes
of the *svg* element.


Fontsizes less than 1 pt are clipped to be 1 pt.
------------------------------------------------

FreeType doesn't allow fonts to get smaller than 1 pt, so all Agg
backends were silently rounding up to 1 pt.  PDF (other vector
backends?) were letting us write fonts that were less than 1 pt, but
they could not be placed properly because position information comes from
FreeType.  This change makes it so no backends can use fonts smaller than
1 pt, consistent with FreeType and ensuring more consistent results across
backends.



Changes to Qt backend class MRO
-------------------------------

To support both Agg and cairo rendering for Qt backends all of the non-Agg
specific code previously in ``backend_qt5agg.FigureCanvasQTAggBase`` has been
moved to ``backend_qt5.FigureCanvasQT`` so it can be shared with the
cairo implementation.  The ``FigureCanvasQTAggBase.paintEvent``,
``FigureCanvasQTAggBase.blit``, and ``FigureCanvasQTAggBase.print_figure``
methods have moved to ``FigureCanvasQTAgg.paintEvent``,
``FigureCanvasQTAgg.blit``, and ``FigureCanvasQTAgg.print_figure``.
The first two methods assume that the instance is also a ``QWidget`` so to use
``FigureCanvasQTAggBase`` it was required to multiple inherit from a
``QWidget`` sub-class.

Having moved all of its methods either up or down the class hierarchy
``FigureCanvasQTAggBase`` has been deprecated.  To do this without warning and
to preserve as much API as possible, ``.backend_qt5agg.FigureCanvasQTAggBase``
now inherits from ``backend_qt5.FigureCanvasQTAgg``.

The MRO for ``FigureCanvasQTAgg`` and ``FigureCanvasQTAggBase`` used to
be ::


   [matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg,
    matplotlib.backends.backend_qt5agg.FigureCanvasQTAggBase,
    matplotlib.backends.backend_agg.FigureCanvasAgg,
    matplotlib.backends.backend_qt5.FigureCanvasQT,
    PyQt5.QtWidgets.QWidget,
    PyQt5.QtCore.QObject,
    sip.wrapper,
    PyQt5.QtGui.QPaintDevice,
    sip.simplewrapper,
    matplotlib.backend_bases.FigureCanvasBase,
    object]

and ::


   [matplotlib.backends.backend_qt5agg.FigureCanvasQTAggBase,
    matplotlib.backends.backend_agg.FigureCanvasAgg,
    matplotlib.backend_bases.FigureCanvasBase,
    object]


respectively.  They are now ::

   [matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg,
    matplotlib.backends.backend_agg.FigureCanvasAgg,
    matplotlib.backends.backend_qt5.FigureCanvasQT,
    PyQt5.QtWidgets.QWidget,
    PyQt5.QtCore.QObject,
    sip.wrapper,
    PyQt5.QtGui.QPaintDevice,
    sip.simplewrapper,
    matplotlib.backend_bases.FigureCanvasBase,
    object]

and ::

   [matplotlib.backends.backend_qt5agg.FigureCanvasQTAggBase,
    matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg,
    matplotlib.backends.backend_agg.FigureCanvasAgg,
    matplotlib.backends.backend_qt5.FigureCanvasQT,
    PyQt5.QtWidgets.QWidget,
    PyQt5.QtCore.QObject,
    sip.wrapper,
    PyQt5.QtGui.QPaintDevice,
    sip.simplewrapper,
    matplotlib.backend_bases.FigureCanvasBase,
    object]




`.axes.Axes.imshow` clips RGB values to the valid range
-------------------------------------------------------

When `.axes.Axes.imshow` is passed an RGB or RGBA value with out-of-range
values, it now logs a warning and clips them to the valid range.
The old behaviour, wrapping back in to the range, often hid outliers
and made interpreting RGB images unreliable.


GTKAgg and GTKCairo backends deprecated
---------------------------------------

The GTKAgg and GTKCairo backends have been deprecated. These obsolete backends
allow figures to be rendered via the GTK+ 2 toolkit. They are untested, known
to be broken, will not work with Python 3, and their use has been discouraged
for some time. Instead, use the ``GTK3Agg`` and ``GTK3Cairo`` backends for
rendering to GTK+ 3 windows.
