Behaviour Changes
-----------------

Tk backend respects file format selection when saving figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When saving a figure from a Tkinter GUI to a filename without an
extension, the file format is now selected based on the value of
the dropdown menu, rather than defaulting to PNG. When the filename
contains an extension, or the OS automatically appends one, the
behavior remains unchanged.

Placing of maximum and minimum minor ticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculation of minor tick locations has been corrected to make the maximum and
minimum minor ticks more consistent.  In some cases this results in an extra
minor tick on an Axis.

``hexbin`` now defaults to ``rcParams["patch.linewidth"]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default value of the *linewidths* argument of `.Axes.hexbin` has
been changed from ``1.0`` to :rc:`patch.linewidth`. This improves the
consistency with `.QuadMesh` in `.Axes.pcolormesh` and `.Axes.hist2d`.

TwoSlopeNorm now auto-expands to always have two slopes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the case where either ``vmin`` or ``vmax`` are not manually specified
to `~.TwoSlopeNorm`, and where the data it is scaling is all less than or
greater than the center point, the limits are now auto-expanded so there
are two symmetrically sized slopes either side of the center point.

Previously ``vmin`` and ``vmax`` were clipped at the center point, which
caused issues when displaying color bars.

This does not affect behaviour when ``vmin`` and ``vmax`` are manually
specified by the user.

Event objects emitted for ``axes_leave_event``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``axes_leave_event`` now emits a synthetic `.LocationEvent`, instead of reusing
the last event object associated with a ``motion_notify_event``.

Streamplot now draws streamlines as one piece if no width or no color variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since there is no need to draw streamlines piece by piece if there is no color
change or width change, now streamplot will draw each streamline in one piece.

The behavior for varying width or varying color is not changed, same logic is
used for these kinds of streamplots.

``canvas`` argument now required for ``FigureFrameWx``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``FigureFrameWx`` now requires a keyword-only ``canvas`` argument
when it is constructed.

``ContourSet`` is now a single Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prior to this release, `.ContourSet` (the object returned by `~.Axes.contour`)
was a custom object holding multiple `.Collection`\s (and not an `.Artist`)
-- one collection per level, each connected component of that level's contour
being an entry in the corresponding collection.

`.ContourSet` is now instead a plain `.Collection` (and thus an `.Artist`).
The collection contains a single path per contour level; this path may be
non-continuous in case there are multiple connected components.

Setting properties on the ContourSet can now usually be done using standard
collection setters (``cset.set_linewidth(3)`` to use the same linewidth
everywhere or ``cset.set_linewidth([1, 2, 3, ...])`` to set different
linewidths on each level) instead of having to go through the individual
sub-components (``cset.collections[0].set_linewidth(...)``).  Note that
during the transition period, it remains possible to access the (deprecated)
``.collections`` attribute; this causes the ContourSet to modify itself to use
the old-style multi-Collection representation.

``SubFigure`` default facecolor is now transparent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subfigures default facecolor changed to ``"none"``. Previously the default was
the value of ``figure.facecolor``.

Reject size related keyword arguments to MovieWriter *grab_frame* method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although we pass `.Figure.savefig` keyword arguments through the
`.AbstractMovieWriter.grab_frame` some of the arguments will result in invalid
output if passed.  To be successfully stitched into a movie, each frame
must be exactly the same size, thus *bbox_inches* and *dpi* are excluded.
Additionally, the movie writers are opinionated about the format of each
frame, so the *format* argument is also excluded.  Passing these
arguments will now raise `TypeError` for all writers (it already did so for some
arguments and some writers).  The *bbox_inches* argument is already ignored (with
a warning) if passed to `.Animation.save`.


Additionally, if :rc:`savefig.bbox` is set to ``'tight'``,
`.AbstractMovieWriter.grab_frame` will now error.  Previously this rcParam
would be temporarily overridden (with a warning) in `.Animation.save`, it is
now additionally overridden in `.AbstractMovieWriter.saving`.

Changes of API after deprecation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `.dviread.find_tex_file` now raises `FileNotFoundError` when the requested filename is
  not found.
- `.Figure.colorbar` now raises if *cax* is not given and it is unable to determine from
  which Axes to steal space, i.e. if *ax* is also not given and *mappable* has not been
  added to an Axes.
- `.pyplot.subplot` and `.pyplot.subplot2grid` no longer auto-remove preexisting
  overlapping Axes; explicitly call ``Axes.remove`` as needed.

Invalid types for Annotation xycoords now raise TypeError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, a `RuntimeError` would be raised in some cases.

Default antialiasing behavior changes for ``Text`` and ``Annotation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``matplotlib.pyplot.annotate()`` and ``matplotlib.pyplot.text()`` now support parameter *antialiased* when initializing.
Examples:

.. code-block:: python

    mpl.text.Text(.5, .5, "foo\nbar", antialiased=True)
    plt.text(0.5, 0.5, '6 inches x 2 inches', antialiased=True)
    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5), antialiased=False)

See "What's New" for more details on usage.

With this new feature, you may want to make sure that you are creating and saving/showing the figure under the same context::

    # previously this was a no-op, now it is what works
    with rccontext(text.antialiased=False):
        fig, ax = plt.subplots()
        ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5))
        fig.savefig('/tmp/test.png')

    # previously this had an effect, now this is a no-op
    fig, ax = plt.subplots()
    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5))
    with rccontext(text.antialiased=False):
        fig.savefig('/tmp/test.png')

Also note that antialiasing for tick labels will be set with :rc:`text.antialiased` when they are created (usually when a ``Figure`` is created) - This means antialiasing for them can no longer be changed by modifying :rc:`text.antialiased`.

``ScalarMappable.to_rgba()`` now respects the mask of RGB(A) arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, the mask was ignored. Now the alpha channel is set to 0 if any
component (R, G, B, or A) is masked.

``Text.get_rotation_mode`` return value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing ``None`` as ``rotation_mode`` to `.Text` (the default value) or passing it to
`.Text.set_rotation_mode` will make `.Text.get_rotation_mode` return ``"default"``
instead of ``None``. The behaviour otherwise is the same.

PostScript paper type adds option to use figure size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :rc:`ps.papertype` rcParam can now be set to ``'figure'``, which will use
a paper size that corresponds exactly with the size of the figure that is being
saved.

``hexbin`` *mincnt* parameter made consistently inclusive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, *mincnt* was inclusive with no *C* provided but exclusive when *C* is provided.
It is now inclusive of *mincnt* in both cases.
