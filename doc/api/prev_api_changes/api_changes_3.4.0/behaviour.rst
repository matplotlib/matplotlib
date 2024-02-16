Behaviour changes
-----------------

Constrained layout rewrite
~~~~~~~~~~~~~~~~~~~~~~~~~~

The layout manager ``constrained_layout`` was re-written with different outer
constraints that should be more robust to complicated subplot layouts.
User-facing changes are:

- some poorly constrained layouts will have different width/height plots than
  before.
- colorbars now respect the ``anchor`` keyword argument of
  `matplotlib.colorbar.make_axes`
- colorbars are wider.
- colorbars in different rows or columns line up more robustly.
- *hspace* and *wspace* options to  `.Figure.set_constrained_layout_pads` were
  twice as wide as the docs said they should be. So these now follow the docs.

This feature will remain "experimental" until the new changes have been used
enough by users, so we anticipate version 3.5 or 3.6. On the other hand,
``constrained_layout`` is extensively tested and used in examples in the
library, so using it should be safe, but layouts may not be exactly the same as
more development takes place.

Details of using ``constrained_layout``, and its algorithm are available at
:ref:`constrainedlayout_guide`

``plt.subplot`` re-selection without keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of `.pyplot.subplot` is to facilitate creating and re-selecting
Axes in a Figure when working strictly in the implicit pyplot API. When
creating new Axes it is possible to select the projection (e.g. polar, 3D, or
various cartographic projections) as well as to pass additional keyword
arguments through to the Axes-subclass that is created.

The first time `.pyplot.subplot` is called for a given position in the Axes
grid it always creates and returns a new Axes with the passed arguments and
projection (defaulting to rectilinear). On subsequent calls to
`.pyplot.subplot` we have to determine if an existing Axes has a) equivalent
parameters, in which case it should be selected as the current Axes and
returned, or b) different parameters, in which case a new Axes is created and
the existing Axes is removed. This leaves the question of what is "equivalent
parameters".

Previously it was the case that an existing Axes subclass, except for Axes3D,
would be considered equivalent to a 2D rectilinear Axes, despite having
different projections, if the keyword arguments (other than *projection*)
matched. Thus::

  ax1 = plt.subplot(1, 1, 1, projection='polar')
  ax2 =  plt.subplots(1, 1, 1)
  ax1 is ax2

We are embracing this long standing behavior to ensure that in the case when no
keyword arguments (of any sort) are passed to `.pyplot.subplot` any existing
Axes is returned, without consideration for keywords or projection used to
initially create it. This will cause a change in behavior when additional
keywords were passed to the original Axes::

  ax1 = plt.subplot(111, projection='polar', theta_offset=.75)
  ax2 = plt.subplots(1, 1, 1)
  ax1 is ax2         # new behavior
  # ax1 is not ax2   # old behavior, made a new axes

  ax1 = plt.subplot(111, label='test')
  ax2 = plt.subplots(1, 1, 1)
  ax1 is ax2         # new behavior
  # ax1 is not ax2   # old behavior, made a new axes

For the same reason, if there was an existing Axes that was not rectilinear,
passing ``projection='rectilinear'`` would reuse the existing Axes ::

  ax1 = plt.subplot(projection='polar')
  ax2 = plt.subplot(projection='rectilinear')
  ax1 is not ax2     # new behavior, makes new Axes
  # ax1 is ax2       # old behavior

contrary to the user's request.

Previously Axes3D could not be re-selected with `.pyplot.subplot` due to an
unrelated bug (also fixed in Matplotlib 3.4). While Axes3D are now consistent
with all other projections there is a change in behavior for ::

  plt.subplot(projection='3d')  # create a 3D Axes

  plt.subplot()                 # now returns existing 3D Axes, but
                                # previously created new 2D Axes

  plt.subplot(projection='rectilinear')  # to get a new 2D Axes

``ioff`` and ``ion`` can be used as context managers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.pyplot.ion` and `.pyplot.ioff` may now be used as context managers to create
a context with interactive mode on or off, respectively. The old behavior of
calling these functions is maintained. To use the new functionality call as::

   with plt.ioff():
      # non-interactive code

Locators and formatters must be in the class hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Axis locators and formatters must now be subclasses of
`~matplotlib.ticker.Locator` and `~matplotlib.ticker.Formatter` respectively.

Date locator for DAILY interval now returns middle of month
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `matplotlib.dates.AutoDateLocator` has a default of
``interval_multiples=True`` that attempts to align ticks with the start of
meaningful intervals like the start of the month, or start of the day, etc.
That lead to approximately 140-day intervals being mapped to the first and 22nd
of the month. This has now been changed so that it chooses the first and 15th
of the month, which is probably what most people want.

``ScalarFormatter`` *useLocale* option obeys grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the `~.ScalarFormatter` option *useLocale* is enabled (or
:rc:`axes.formatter.use_locale` is *True*) and the configured locale uses
grouping, a separator will be added as described in `locale.format_string`.

``Axes.errorbar`` cycles non-color properties correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly, `.Axes.errorbar` incorrectly skipped the Axes property cycle if a
color was explicitly specified, even if the property cycler was for other
properties (such as line style). Now, `.Axes.errorbar` will advance the Axes
property cycle as done for `.Axes.plot`, i.e., as long as all properties in the
cycler are not explicitly passed.

pyplot.specgram always uses origin='upper'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously if :rc:`image.origin` was set to something other than ``'upper'`` or
if the *origin* keyword argument was passed with a value other than
``'upper'``, the spectrogram itself would flip, but the Axes would remain
oriented for an origin value of ``'upper'``, so that the resulting plot was
incorrectly labelled.

Now, the *origin* keyword argument is not supported and the ``image.origin``
rcParam is ignored. The function `matplotlib.pyplot.specgram` is forced to use
``origin='upper'``, so that the Axes are correct for the plotted spectrogram.

xunits=None and yunits=None passed as keyword arguments are treated as "no action"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many (but not all) of the methods on `~.axes.Axes` take the (undocumented)
keyword arguments *xunits* and *yunits* that will update the units on the given
Axis by calling `.Axis.set_units` and `.Axis.update_units`.

Previously if *None* was passed it would clear the value stored in
``.Axis.units`` which will in turn break converters which rely on the value in
``.Axis.units`` to work properly (notably `.StrCategoryConverter`).

This changes the semantics of ``ax.meth(..., xunits=None, yunits=None)`` from
"please clear the units" to "do the default thing as if they had not been
passed" which is consistent with the standard behavior of Matplotlib keyword
arguments.

If you were relying on passing ``xunits=None`` to plotting methods to clear the
``.Axes.units`` attribute, directly call `.Axis.set_units` (and
`.Axis.update_units` if you also require the converter to be updated).

Annotations with ``annotation_clip`` no longer affect ``tight_layout``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, `.text.Annotation.get_tightbbox` always returned the full
`.text.Annotation.get_window_extent` of the object, independent of the value of
``annotation_clip``. `.text.Annotation.get_tightbbox` now correctly takes this
extra clipping box into account, meaning that `~.text.Annotation`\s that are
not drawn because of ``annotation_clip`` will not count towards the Axes
bounding box calculations, such as those done by `~.pyplot.tight_layout`.

This is now consistent with the API described in `~.artist.Artist`, which
specifies that ``get_window_extent`` should return the full extents and
``get_tightbbox`` should "account for any clipping".

Parasite Axes pcolor and pcolormesh now defaults to placing grid edges at integers, not half-integers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is consistent with `~.Axes.pcolor` and `~.Axes.pcolormesh`.

``Colorbar`` outline is now a ``Spine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The outline of `~matplotlib.colorbar.Colorbar` is now a `.Spine` and drawn as
one, instead of a `.Polygon` drawn as an artist. This ensures it will always be
drawn after (i.e., on top of) all artists, consistent with Spines on normal
Axes.

``Colorbar.dividers`` changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This attribute is now always a `.LineCollection` -- an empty one if
``drawedges`` is *False*. Its default colors and linewidth
(:rc:`axes.edgecolor`, :rc:`axes.linewidth`) are now resolved at instantiation
time, not at draw time.

Raise or warn on registering a colormap twice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``matplotlib.cm.register_cmap`` to register a user provided or
third-party colormap it will now raise a `ValueError` if trying to over-write
one of the built in colormaps and warn if trying to over write a user
registered colormap. This may raise for user-registered colormaps in the
future.

Consecutive rasterized draws now merged
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tracking of depth of raster draws has moved from
`.backend_mixed.MixedModeRenderer.start_rasterizing` and
`.backend_mixed.MixedModeRenderer.stop_rasterizing` into
`.artist.allow_rasterization`. This means the start and stop functions are only
called when the rasterization actually needs to be started and stopped.

The output of vector backends will change in the case that rasterized elements
are merged. This should not change the appearance of outputs.

The renders in 3rd party backends are now expected to have
``self._raster_depth`` and ``self._rasterizing`` initialized to ``0`` and
*False* respectively.

Consistent behavior of ``draw_if_interactive()`` across backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.pyplot.draw_if_interactive` no longer shows the window (if it was previously
unshown) on the Tk and nbAgg backends, consistently with all other backends.

The Artist property *rasterized* cannot be *None* anymore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is now a boolean only. Before the default was *None* and
`.Artist.set_rasterized` was documented to accept *None*. However, *None* did
not have a special meaning and was treated as *False*.

Canvas's callback registry now stored on Figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical location of the `~.cbook.CallbackRegistry` used to handle
Figure/Canvas events has been moved from the Canvas to the Figure. This change
should be transparent to almost all users, however if you are swapping
switching the Figure out from on top of a Canvas or visa versa you may see a
change in behavior.

Harmonized key event data across backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The different backends with key translation support, now handle "Shift" as a
sometimes modifier, where the ``'shift+'`` prefix won't be added if a key
translation was made.

In the Qt5 backend, the ``matplotlib.backends.backend_qt5.SPECIAL_KEYS``
dictionary contains keys that do *not* return their unicode name instead they
have manually specified names. The name for ``QtCore.Qt.Key_Meta`` has changed
to ``'meta'`` to be consistent with the other GUI backends.

The WebAgg backend now handles key translations correctly on non-US keyboard
layouts.

In the GTK and Tk backends, the handling of non-ASCII keypresses (as reported
in the KeyEvent passed to ``key_press_event``-handlers) now correctly reports
Unicode characters (e.g., €), and better respects NumLock on the numpad.

In the GTK and Tk backends, the following key names have changed; the new names
are consistent with those reported by the Qt backends:

- The "Break/Pause" key (keysym 0xff13) is now reported as ``"pause"`` instead
  of ``"break"`` (this is also consistent with the X key name).
- The numpad "delete" key is now reported as ``"delete"`` instead of ``"dec"``.

WebAgg backend no longer reports a middle click as a right click
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously when using the WebAgg backend the event passed to a callback by
``fig.canvas.mpl_connect('mouse_button_event', callback)`` on a middle click
would report `.MouseButton.RIGHT` instead of `.MouseButton.MIDDLE`.

ID attribute of XML tags in SVG files now based on SHA256 rather than MD5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib generates unique ID attributes for various tags in SVG files.
Matplotlib previously generated these unique IDs using the first 10 characters
of an MD5 hash. The MD5 hashing algorithm is not available in Python on systems
with Federal Information Processing Standards (FIPS) enabled. Matplotlib now
uses the first 10 characters of an SHA256 hash instead. SVG files that would
otherwise match those saved with earlier versions of matplotlib, will have
different ID attributes.

``RendererPS.set_font`` is no longer a no-op in AFM mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.RendererPS.set_font` now sets the current PostScript font in all cases.

Autoscaling in Axes3D
~~~~~~~~~~~~~~~~~~~~~

In Matplotlib 3.2.0, autoscaling was made lazier for 2D Axes, i.e., limits
would only be recomputed when actually rendering the canvas, or when the user
queries the Axes limits. This performance improvement is now extended to
`.Axes3D`. This also fixes some issues with autoscaling being triggered
unexpectedly in Axes3D.

Please see :ref:`the API change for 2D Axes <api-changes-3-2-0-autoscaling>`
for further details.

Axes3D automatically adding itself to Figure is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New `.Axes3D` objects previously added themselves to figures when they were
created, unlike all other Axes classes, which lead to them being added twice if
``fig.add_subplot(111, projection='3d')`` was called.

This behavior is now deprecated and will warn. The new keyword argument
*auto_add_to_figure* controls the behavior and can be used to suppress the
warning. The default value will change to *False* in Matplotlib 3.5, and any
non-*False* value will be an error in Matplotlib 3.6.

In the future, `.Axes3D` will need to be explicitly added to the figure ::

  fig = Figure()
  # create Axes3D
  ax = Axes3d(fig)
  # add to Figure
  fig.add_axes(ax)

as needs to be done for other `.axes.Axes` sub-classes. Or, a 3D projection can
be made via::

    fig.add_subplot(projection='3d')

``mplot3d.art3d.get_dir_vector`` always returns NumPy arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For consistency, `~.mplot3d.art3d.get_dir_vector` now always returns NumPy
arrays, even if the input is a 3-element iterable.

Changed cursive and fantasy font definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Comic Sans and Comic Neue fonts were moved from the default
:rc:`font.fantasy` list to the default :rc:`font.cursive` setting, in
accordance with the CSS font families example_ and in order to provide a
cursive font present in Microsoft's Core Fonts set.

.. _example: https://www.w3.org/Style/Examples/007/fonts.en.html

docstring.Substitution now always dedents docstrings before string interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
