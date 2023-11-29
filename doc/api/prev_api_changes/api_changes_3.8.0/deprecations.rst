Deprecations
------------

Calling ``paths.get_path_collection_extents`` with empty *offsets*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling  `~.get_path_collection_extents` with an empty *offsets* parameter
has an ambiguous interpretation and is therefore deprecated. When the
deprecation period expires, this will produce an error.


``axes_grid1.axes_divider`` API changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``AxesLocator`` class is deprecated.  The ``new_locator`` method of divider
instances now instead returns an opaque callable (which can still be passed to
``ax.set_axes_locator``).

``Divider.locate`` is deprecated; use ``Divider.new_locator(...)(ax, renderer)``
instead.


``bbox.anchored()`` with no explicit container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not passing a *container* argument to `.BboxBase.anchored` is now deprecated.


Functions in ``mpl_toolkits.mplot3d.proj3d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function ``transform`` is just an alias for ``proj_transform``,
use the latter instead.

The following functions are either unused (so no longer required in Matplotlib)
or considered private. If you rely on them, please make a copy of the code,
including all functions that starts with a ``_`` (considered private).

* ``ortho_transformation``
* ``persp_transformation``
* ``proj_points``
* ``proj_trans_points``
* ``rot_x``
* ``rotation_about_vector``
* ``view_transformation``


Arguments other than ``renderer`` to ``get_tightbbox``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are keyword-only arguments. This is for consistency and that
different classes have different additional arguments.


The object returned by ``pcolor()`` has changed to a ``PolyQuadMesh`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The old object was a `.PolyCollection` with flattened vertices and array data.
The new `.PolyQuadMesh` class subclasses `.PolyCollection`, but adds in better
2D coordinate and array handling in alignment with `.QuadMesh`. Previously, if
a masked array was input, the list of polygons within the collection would shrink
to the size of valid polygons and users were required to keep track of which
polygons were drawn and call ``set_array()`` with the smaller "compressed" array size.
Passing the "compressed" and flattened array values is now deprecated and the
full 2D array of values (including the mask) should be passed
to `.PolyQuadMesh.set_array`.


``LocationEvent.lastevent``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated with no replacement.


``allsegs``, ``allkinds``, ``tcolors`` and ``tlinewidths`` attributes of `.ContourSet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated; if required, directly retrieve the vertices
and codes of the Path objects from ``ContourSet.get_paths()`` and the colors
and the linewidths via ``ContourSet.get_facecolor()``, ``ContourSet.get_edgecolor()``
and ``ContourSet.get_linewidths()``.


``ContourSet.collections``
~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.  `.ContourSet` is now implemented as a single `.Collection` of paths,
each path corresponding to a contour level, possibly including multiple unconnected
components.

During the deprecation period, accessing ``ContourSet.collections`` will revert the
current ContourSet instance to the old object layout, with a separate `.PathCollection`
per contour level.


``INVALID_NON_AFFINE``, ``INVALID_AFFINE``, ``INVALID`` attributes of ``TransformNode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated.


``Grouper.clean()``
~~~~~~~~~~~~~~~~~~~

with no replacement. The Grouper class now cleans itself up automatically.


``GridHelperCurveLinear.get_data_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.  Use ``grid_finder.extreme_finder(*[None] * 5)`` to get the
extremes of the grid.


*np_load* parameter of ``cbook.get_sample_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This parameter is deprecated; `.get_sample_data` now auto-loads numpy arrays.
Use ``get_sample_data(..., asfileobj=False)`` instead to get the filename of
the data file, which can then be passed to `open`, if desired.


``RendererAgg.tostring_rgb`` and ``FigureCanvasAgg.tostring_rgb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are deprecated with no direct replacement. Consider using ``buffer_rgba``
instead, which should cover most use cases.


The parameter of ``Annotation.contains`` and ``Legend.contains`` is renamed to *mouseevent*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... consistently with `.Artist.contains`.


Accessing ``event.guiEvent`` after event handlers return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated: for some GUI toolkits, it is unsafe to do so.  In the
future, ``event.guiEvent`` will be set to None once the event handlers return;
you may separately stash the object at your own risk.


Widgets
~~~~~~~

The *visible* attribute getter of Selector widgets has been deprecated;
use ``get_visible``


Method parameters renamed to match base classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The only parameter of ``transform_affine`` and ``transform_non_affine`` in ``Transform`` subclasses is renamed
to *values*.

The *points* parameter of ``transforms.IdentityTransform.transform`` is renamed to *values*.

The *trans* parameter of ``table.Cell.set_transform`` is renamed to *t* consistently with
`.Artist.set_transform`.

The *clippath* parameters of ``axis.Axis.set_clip_path``  and ``axis.Tick.set_clip_path`` are
renamed to *path* consistently with `.Artist.set_clip_path`.

The *s* parameter of ``images.NonUniformImage.set_filternorm`` is renamed to *filternorm*
consistently with ```_ImageBase.set_filternorm``.

The *s* parameter of ``images.NonUniformImage.set_filterrad`` is renamed to *filterrad*
consistently with ```_ImageBase.set_filterrad``.


*numdecs* parameter and attribute of ``LogLocator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are deprecated without replacement, because they have no effect.


``NavigationToolbar2QT.message`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... with no replacement.


``ft2font.FT2Image.draw_rect`` and ``ft2font.FT2Font.get_xys``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated as they are unused. If you rely on these, please let us know.


``backend_ps.psDefs``
~~~~~~~~~~~~~~~~~~~~~

The ``psDefs`` module-level variable in ``backend_ps`` is deprecated with no
replacement.


Callable axisartist Axes
~~~~~~~~~~~~~~~~~~~~~~~~
Calling an axisartist Axes to mean `~matplotlib.pyplot.axis` is deprecated; explicitly
call the method instead.


``AnchoredEllipse`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instead, directly construct an `.AnchoredOffsetbox`, an `.AuxTransformBox`, and an
`~.patches.Ellipse`, as demonstrated in :doc:`/gallery/misc/anchored_artists`.


Automatic papersize selection in PostScript
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting :rc:`ps.papersize` to ``'auto'`` or passing ``papersize='auto'`` to
`.Figure.savefig` is deprecated. Either pass an explicit paper type name, or
omit this parameter to use the default from the rcParam.


``Tick.set_label1`` and ``Tick.set_label2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are deprecated.  Calling these methods from third-party code usually has no
effect, as the labels are overwritten at draw time by the tick formatter.


Passing extra positional arguments to ``Figure.add_axes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Positional arguments passed to `.Figure.add_axes` other than a rect or an
existing ``Axes`` are currently ignored, and doing so is now deprecated.


``CbarAxesBase.toggle_label``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.  Instead, use standard methods for manipulating colorbar
labels (`.Colorbar.set_label`) and tick labels (`.Axes.tick_params`).


``TexManager.texcache``
~~~~~~~~~~~~~~~~~~~~~~~

... is considered private and deprecated. The location of the cache directory is
clarified in the doc-string.


Artists explicitly passed in will no longer be filtered by legend() based on their label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, artists explicitly passed to ``legend(handles=[...])`` are filtered
out if their label starts with an underscore.  This behavior is deprecated;
explicitly filter out such artists
(``[art for art in artists if not art.get_label().startswith('_')]``) if
necessary.


``FigureCanvasBase.switch_backends``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated with no replacement.


``cbook.Stack`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... with no replacement.


``inset_location.InsetPosition`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use `~.Axes.inset_axes` instead.


``axisartist.axes_grid`` and ``axisartist.axes_rgb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These modules, which provide wrappers combining the functionality of
`.axes_grid1` and `.axisartist`, are deprecated; directly use e.g.
``AxesGrid(..., axes_class=axislines.Axes)`` instead.


``ContourSet.antialiased``
~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated; use `~.Collection.get_antialiased` or
`~.Collection.set_antialiased` instead.  Note that `~.Collection.get_antialiased`
returns an array.


Passing non-int or sequence of non-int to ``Table.auto_set_column_width``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Column numbers are ints, and formerly passing any other type was effectively
ignored. This will become an error in the future.


``PdfPages(keep_empty=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A zero-page pdf is not valid, thus passing ``keep_empty=True`` to
`.backend_pdf.PdfPages` and `.backend_pgf.PdfPages`, and the ``keep_empty``
attribute of these classes, are deprecated.  Currently, these classes default
to keeping empty outputs, but that behavior is deprecated too.  Explicitly
passing ``keep_empty=False`` remains supported for now to help transition to
the new behavior.

Furthermore, `.backend_pdf.PdfPages` no longer immediately creates the target
file upon instantiation, but only when the first figure is saved.  To fully
control file creation, directly pass an opened file object as argument
(``with open(path, "wb") as file, PdfPages(file) as pdf: ...``).


Auto-closing of figures when switching backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.  Explicitly call ``plt.close("all")`` if necessary.  In the
future, allowable backend switches (i.e. those that do not swap a GUI event
loop with another one) will not close existing figures.


Support for passing the "frac" key in ``annotate(..., arrowprops={"frac": ...})``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... has been removed.  This key has had no effect since Matplotlib 1.5.
