Removals
--------


ttconv removed
~~~~~~~~~~~~~~

The ``matplotlib._ttconv`` extension has been removed. Most of its
functionaliy was already replaced by other code, and the only thing left
was embedding TTF fonts in PostScript in Type 42 format. This is now
done in the PS backend using the FontTools library.

Remove hard reference to ``lastevent`` in ``LocationEvent``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This was previously used to detect exiting from axes, however the hard
reference would keep closed `.Figure` objects and their children alive longer
than expected.

``ft2font.FT2Image.draw_rect`` and ``ft2font.FT2Font.get_xys``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... have been removed as they are unused.

``Tick.set_label``, ``Tick.set_label1`` and ``Tick.set_label2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are removed.  Calling these methods from third-party code usually had no
effect, as the labels are overwritten at draw time by the tick formatter.


Functions in ``mpl_toolkits.mplot3d.proj3d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function ``transform`` is just an alias for ``proj_transform``,
use the latter instead.

The following functions were either unused (so no longer required in Matplotlib)
or considered private.

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
consistently with ``_ImageBase.set_filternorm``.

The *s* parameter of ``images.NonUniformImage.set_filterrad`` is renamed to *filterrad*
consistently with ``_ImageBase.set_filterrad``.

The only parameter of ``Annotation.contains`` and ``Legend.contains`` is renamed to *mouseevent*
consistently with `.Artist.contains`.

Method parameters renamed
~~~~~~~~~~~~~~~~~~~~~~~~~

The *p* parameter of ``BboxBase.padded`` is renamed to *w_pad*, consistently with the other parameter, *h_pad*

*numdecs* parameter and attribute of ``LogLocator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are removed without replacement, because they had no effect.
The ``PolyQuadMesh`` class requires full 2D arrays of values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, if a masked array was input, the list of polygons within the collection
would shrink to the size of valid polygons and users were required to keep track of
which polygons were drawn and call ``set_array()`` with the smaller "compressed"
array size. Passing the "compressed" and flattened array values will no longer
work and the full 2D array of values (including the mask) should be passed
to `.PolyQuadMesh.set_array`.
``ContourSet.collections``
~~~~~~~~~~~~~~~~~~~~~~~~~~

... has been removed.  `~.ContourSet` is now implemented as a single
`~.Collection` of paths, each path corresponding to a contour level, possibly
including multiple unconnected components.

``ContourSet.antialiased``
~~~~~~~~~~~~~~~~~~~~~~~~~~

... has been removed.  Use `~.Collection.get_antialiased` or
`~.Collection.set_antialiased` instead.  Note that `~.Collection.get_antialiased`
returns an array.

``tcolors`` and ``tlinewidths`` attributes of ``ContourSet``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... have been removed.  Use `~.Collection.get_facecolor`, `~.Collection.get_edgecolor`
or `~.Collection.get_linewidths` instead.


``calc_label_rot_and_inline`` method of ``ContourLabeler``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... has been removed without replacement.


``add_label_clabeltext`` method of ``ContourLabeler``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... has been removed.  Use `~.ContourLabeler.add_label` instead.
Passing extra positional arguments to ``Figure.add_axes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Positional arguments passed to `.Figure.add_axes` other than a rect or an existing
``Axes`` were previously ignored, and is now an error.


Artists explicitly passed in will no longer be filtered by legend() based on their label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, artists explicitly passed to ``legend(handles=[...])`` are filtered out if
their label starts with an underscore. This filter is no longer applied; explicitly
filter out such artists (``[art for art in artists if not
art.get_label().startswith('_')]``) if necessary.

Note that if no handles are specified at all, then the default still filters out labels
starting with an underscore.


The parameter of ``Annotation.contains`` and ``Legend.contains`` is renamed to *mouseevent*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... consistently with `.Artist.contains`.


Support for passing the "frac" key in ``annotate(..., arrowprops={"frac": ...})``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... has been removed.  This key has had no effect since Matplotlib 1.5.


Passing non-int or sequence of non-int to ``Table.auto_set_column_width``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Column numbers are ints, and formerly passing any other type was effectively ignored.
This has now become an error.


Widgets
~~~~~~~

The *visible* attribute getter of ``*Selector`` widgets has been removed; use
``get_visible`` instead.


Auto-closing of figures when switching backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allowable backend switches (i.e. those that do not swap a GUI event loop with another
one) will not close existing figures. If necessary, call ``plt.close("all")`` before
switching.


``FigureCanvasBase.switch_backends``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... has been removed with no replacement.


Accessing ``event.guiEvent`` after event handlers return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is no longer supported, and ``event.guiEvent`` will be set to None once the event
handlers return. For some GUI toolkits, it is unsafe to use the event, though you may
separately stash the object at your own risk.


``PdfPages(keep_empty=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A zero-page PDF is not valid, thus passing ``keep_empty=True`` to `.backend_pdf.PdfPages`
and `.backend_pgf.PdfPages`, and the ``keep_empty`` attribute of these classes, is no
longer allowed, and empty PDF files will not be created.

Furthermore, `.backend_pdf.PdfPages` no longer immediately creates the target file upon
instantiation, but only when the first figure is saved.  To fully control file creation,
directly pass an opened file object as argument (``with open(path, "wb") as file,
PdfPages(file) as pdf: ...``).


``backend_ps.psDefs``
~~~~~~~~~~~~~~~~~~~~~

The ``psDefs`` module-level variable in ``backend_ps`` has been removed with no
replacement.


Automatic papersize selection in PostScript
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting :rc:`ps.papersize` to ``'auto'`` or passing ``papersize='auto'`` to
`.Figure.savefig` is no longer supported. Either pass an explicit paper type name, or
omit this parameter to use the default from the rcParam.


``RendererAgg.tostring_rgb`` and ``FigureCanvasAgg.tostring_rgb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... have been remove with no direct replacement. Consider using ``buffer_rgba`` instead,
which should cover most use cases.


``NavigationToolbar2QT.message`` has been removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... with no replacement.


``TexManager.texcache``
~~~~~~~~~~~~~~~~~~~~~~~

... is considered private and has been removed. The location of the cache directory is
clarified in the doc-string.


``cbook`` API changes
~~~~~~~~~~~~~~~~~~~~~

``cbook.Stack`` has been removed with no replacement.

``Grouper.clean()`` has been removed with no replacement. The Grouper class now cleans
itself up automatically.

The *np_load* parameter of ``cbook.get_sample_data`` has been removed; `.get_sample_data`
now auto-loads numpy arrays. Use ``get_sample_data(..., asfileobj=False)`` instead to get
the filename of the data file, which can then be passed to `open`, if desired.


Calling ``paths.get_path_collection_extents`` with empty *offsets*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling  `~.get_path_collection_extents` with an empty *offsets* parameter has an
ambiguous interpretation and is no longer allowed.


``bbox.anchored()`` with no explicit container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not passing a *container* argument to `.BboxBase.anchored` is no longer supported.


``INVALID_NON_AFFINE``, ``INVALID_AFFINE``, ``INVALID`` attributes of ``TransformNode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These attributes have been removed.


``axes_grid1`` API changes
~~~~~~~~~~~~~~~~~~~~~~~~~~

``anchored_artists.AnchoredEllipse`` has been removed. Instead, directly construct an
`.AnchoredOffsetbox`, an `.AuxTransformBox`, and an `~.patches.Ellipse`, as demonstrated
in :doc:`/gallery/misc/anchored_artists`.

The ``axes_divider.AxesLocator`` class has been removed.  The ``new_locator`` method of
divider instances now instead returns an opaque callable (which can still be passed to
``ax.set_axes_locator``).

``axes_divider.Divider.locate`` has been removed; use ``Divider.new_locator(...)(ax,
renderer)`` instead.

``axes_grid.CbarAxesBase.toggle_label`` has been removed. Instead, use standard methods
for manipulating colorbar labels (`.Colorbar.set_label`) and tick labels
(`.Axes.tick_params`).

``inset_location.InsetPosition`` has been removed; use `~.Axes.inset_axes` instead.


``axisartist`` API changes
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``axisartist.axes_grid`` and ``axisartist.axes_rgb`` modules, which provide wrappers
combining the functionality of `.axes_grid1` and `.axisartist`, have been removed;
directly use e.g. ``AxesGrid(..., axes_class=axislines.Axes)`` instead.

Calling an axisartist Axes to mean `~matplotlib.pyplot.axis` has been removed; explicitly
call the method instead.

``floating_axes.GridHelperCurveLinear.get_data_boundary`` has been removed.  Use
``grid_finder.extreme_finder(*[None] * 5)`` to get the extremes of the grid.
