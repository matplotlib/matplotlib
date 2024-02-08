Behaviour changes
-----------------

``plt.get_cmap`` and ``matplotlib.cm.get_cmap`` return a copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly, `~.pyplot.get_cmap` and ``matplotlib.cm.get_cmap`` returned a global version
of a `.Colormap`. This was prone to errors as modification of the colormap would
propagate from one location to another without warning. Now, a new copy of the colormap
is returned.

Large ``imshow`` images are now downsampled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When showing an image using `~matplotlib.axes.Axes.imshow` that has more than
:math:`2^{24}` columns or :math:`2^{23}` rows, the image will now be
downsampled to below this resolution before being resampled for display by the
AGG renderer. Previously such a large image would be shown incorrectly. To
prevent this downsampling and the warning it raises, manually downsample your
data before handing it to `~matplotlib.axes.Axes.imshow`.

Default date limits changed to 1970-01-01 – 1970-01-02
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously the default limits for an empty axis set up for dates
(`.Axis.axis_date`) was 2000-01-01 to 2010-01-01. This has been changed to
1970-01-01 to 1970-01-02. With the default epoch, this makes the numeric limit
for date axes the same as for other axes (0.0-1.0), and users are less likely
to set a locator with far too many ticks.

*markerfmt* argument to ``stem``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The behavior of the *markerfmt* parameter of `~.Axes.stem` has changed:

- If *markerfmt* does not contain a color, the color is taken from *linefmt*.
- If *markerfmt* does not contain a marker, the default is 'o'.

Before, *markerfmt* was passed unmodified to ``plot(..., fmt)``, which had a
number of unintended side-effects; e.g. only giving a color switched to a solid
line without markers.

For a simple call ``stem(x, y)`` without parameters, the new rules still
reproduce the old behavior.

``get_ticklabels`` now always populates labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously `.Axis.get_ticklabels` (and `.Axes.get_xticklabels`,
`.Axes.get_yticklabels`) would only return empty strings unless a draw had
already been performed. Now the ticks and their labels are updated when the
labels are requested.

Warning when scatter plot color settings discarded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When making an animation of a scatter plot, if you don't set *c* (the color
value parameter) when initializing the artist, the color settings are ignored.
`.Axes.scatter` now raises a warning if color-related settings are changed
without setting *c*.

3D ``contourf`` polygons placed between levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The polygons used in a 3D `~.Axes3D.contourf` plot are now placed halfway
between the contour levels, as each polygon represents the location of values
that lie between two levels.

Axes title now avoids y-axis offset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, Axes titles could overlap the y-axis offset text, which is often in
the upper left corner of the axes. Now titles are moved above the offset text
if overlapping when automatic title positioning is in effect (i.e. if *y* in
`.Axes.set_title` is *None* and :rc:`axes.titley` is also *None*).

Dotted operators gain extra space in mathtext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In mathtext, ``\doteq \doteqdot \dotminus \dotplus \dots`` are now surrounded
by extra space because they are correctly treated as relational or binary
operators.

*math* parameter of ``mathtext.get_unicode_index`` defaults to False
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In math mode, ASCII hyphens (U+002D) are now replaced by Unicode minus signs
(U+2212) at the parsing stage.

``ArtistList`` proxies copy contents on iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When iterating over the contents of the dynamically generated proxy lists for
the Artist-type accessors (see :ref:`Behavioural API Changes 3.5 - Axes
children combined`), a copy of the contents is made. This ensure that artists
can safely be added or removed from the Axes while iterating over their
children.

This is a departure from the expected behavior of mutable iterable data types
in Python — iterating over a list while mutating it has surprising consequences
and dictionaries will error if they change size during iteration. Because all
of the accessors are filtered views of the same underlying list, it is possible
for seemingly unrelated changes, such as removing a Line, to affect the
iteration over any of the other accessors. In this case, we have opted to make
a copy of the relevant children before yielding them to the user.

This change is also consistent with our plan to make these accessors immutable
in Matplotlib 3.7.

``AxesImage`` string representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The string representation of `.AxesImage` changes from stating the position in
the figure ``"AxesImage(80,52.8;496x369.6)"`` to giving the number of pixels
``"AxesImage(size=(300, 200))"``.

Improved autoscaling for Bézier curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bézier curves are now autoscaled to their extents - previously they were
autoscaled to their ends and control points, which in some cases led to
unnecessarily large limits.

``QuadMesh`` mouseover defaults to False
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New in 3.5, `.QuadMesh.get_cursor_data` allows display of data values under the
cursor. However, this can be very slow for large meshes, so mouseover now
defaults to *False*.

Changed pgf backend document class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pgf backend now uses the ``article`` document class as basis for
compilation.

``MathtextBackendAgg.get_results`` no longer returns ``used_characters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last item (``used_characters``) in the tuple returned by
``MathtextBackendAgg.get_results`` has been removed. In order to unpack this
tuple in a backward and forward-compatible way, use e.g. ``ox, oy, width,
height, descent, image, *_ = parse(...)``, which will ignore
``used_characters`` if it was present.

``Type1Font`` objects include more properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``matplotlib._type1font.Type1Font.prop`` dictionary now includes more keys,
such as ``CharStrings`` and ``Subrs``. The value of the ``Encoding`` key is now
a dictionary mapping codes to glyph names. The
``matplotlib._type1font.Type1Font.transform`` method now correctly removes
``UniqueID`` properties from the font.

``rcParams.copy()`` returns ``RcParams`` rather than ``dict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returning an `.RcParams` instance from `.RcParams.copy` makes the copy still
validate inputs, and additionally avoids emitting deprecation warnings when
using a previously copied instance to update the global instance (even if some
entries are deprecated).

``rc_context`` no longer resets the value of ``'backend'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`matplotlib.rc_context` incorrectly reset the value of :rc:`backend` if backend
resolution was triggered in the context. This affected only the value. The
actual backend was not changed. Now, `matplotlib.rc_context` does not reset
:rc:`backend` anymore.

Default ``rcParams["animation.convert_args"]`` changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It now defaults to ``["-layers", "OptimizePlus"]`` to try to generate smaller
GIFs. Set it back to an empty list to recover the previous behavior.

Style file encoding now specified to be UTF-8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It has been impossible to import Matplotlib with a non UTF-8 compatible locale
encoding because we read the style library at import time. This change is
formalizing and documenting the status quo so there is no deprecation period.

MacOSX backend uses sRGB instead of GenericRGB color space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MacOSX backend now display sRGB tagged image instead of GenericRGB which is an
older (now deprecated) Apple color space. This is the source color space used
by ColorSync to convert to the current display profile.

Renderer optional for ``get_tightbbox`` and ``get_window_extent``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.Artist.get_tightbbox` and `.Artist.get_window_extent` methods no longer
require the *renderer* keyword argument, saving users from having to query it
from ``fig.canvas.get_renderer``. If the *renderer* keyword argument is not
supplied, these methods first check if there is a cached renderer from a
previous draw and use that. If there is no cached renderer, then the methods
will use ``fig.canvas.get_renderer()`` as a fallback.

``FigureFrameWx`` constructor, subclasses, and ``get_canvas``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FigureCanvasWx`` constructor gained a *canvas_class* keyword-only
parameter which specifies the canvas class that should be used. This parameter
will become required in the future. The ``get_canvas`` method, which was
previously used to customize canvas creation, is deprecated. The
``FigureFrameWxAgg`` and ``FigureFrameWxCairo`` subclasses, which overrode
``get_canvas``, are deprecated.

``FigureFrameWx.sizer``
~~~~~~~~~~~~~~~~~~~~~~~

... has been removed. The frame layout is no longer based on a sizer, as the
canvas is now the sole child widget; the toolbar is now a regular toolbar added
using ``SetToolBar``.

Incompatible layout engines raise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You cannot switch between ``tight_layout`` and ``constrained_layout`` if a
colorbar has already been added to a figure. Invoking the incompatible layout
engine used to warn, but now raises with a `RuntimeError`.

``CallbackRegistry`` raises on unknown signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When Matplotlib instantiates a `.CallbackRegistry`, it now limits callbacks to
the signals that the registry knows about. In practice, this means that calling
`~.FigureCanvasBase.mpl_connect` with an invalid signal name now raises a
`ValueError`.

Changed exception type for incorrect SVG date metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Providing date metadata with incorrect type to the SVG backend earlier resulted
in a `ValueError`. Now, a `TypeError` is raised instead.

Specified exception types in ``Grid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a few cases an `Exception` was thrown when an incorrect argument value was
set in the `mpl_toolkits.axes_grid1.axes_grid.Grid` (=
`mpl_toolkits.axisartist.axes_grid.Grid`) constructor. These are replaced as
follows:

* Providing an incorrect value for *ngrids* now raises a `ValueError`
* Providing an incorrect type for *rect* now raises a `TypeError`
