
Behavior changes
----------------

Reduced default value of :rc:`axes.formatter.limits`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed the default value of :rc:`axes.formatter.limits` from -7, 7 to
-5, 6 for better readability.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   fig, (ax_old, ax_new) = plt.subplots(1, 2, constrained_layout=True)

   ax_new.set_title('new values (-5, 6)')
   ax_old.set_title('old values (-7, 7)')

   x = np.logspace(-8, 8, 1024)
   y = 1e-5 * np.exp(-x / 1e5) + 1e-6

   ax_old.xaxis.get_major_formatter().set_powerlimits((-7, 7))
   ax_old.yaxis.get_major_formatter().set_powerlimits((-7, 7))

   for ax in [ax_new, ax_old]:
       ax.plot(x, y)
       ax.set_xlim(0, 1e6)
       ax.set_ylim(1e-6, 1e-5)


`matplotlib.colorbar.Colorbar` uses un-normalized axes for all mappables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before 3.0, `matplotlib.colorbar.Colorbar` (`~.Figure.colorbar`) normalized
all axes limits between 0 and 1 and had custom tickers to handle the
labelling of the colorbar ticks.  After 3.0, colorbars constructed from
mappables that were *not* contours were constructed with axes that had
limits between ``vmin`` and ``vmax`` of the mappable's norm, and the tickers
were made children of the normal axes tickers.

This version of Matplotlib extends that to mappables made by contours, and
allows the axes to run between the lowest boundary in the contour and the
highest.

Code that worked around the normalization between 0 and 1 will need to be
modified.

``MovieWriterRegistry``
~~~~~~~~~~~~~~~~~~~~~~~
`.MovieWriterRegistry` now always checks the availability of the writer classes
before returning them.  If one wishes, for example, to get the first available
writer, without performing the availability check on subsequent writers, it is
now possible to iterate over the registry, which will yield the names of the
available classes.

.. _api-changes-3-2-0-autoscaling:

Autoscaling
~~~~~~~~~~~

Matplotlib used to recompute autoscaled limits after every plotting
(``plot()``, ``bar()``, etc.) call.  It now only does so when actually
rendering the canvas, or when the user queries the Axes limits.  This is a
major performance improvement for plots with a large number of artists.

In particular, this means that artists added manually with `.Axes.add_line`,
`.Axes.add_patch`, etc. will be taken into account by the autoscale, even
without an explicit call to `.Axes.autoscale_view`.

In some cases, this can result in different limits being reported.  If this is
an issue, consider triggering a draw with ``fig.canvas.draw()``.

Autoscaling has also changed for artists that are based on the `.Collection`
class.  Previously, the method that calculates the automatic limits
`.Collection.get_datalim` tried to take into account the size of objects
in the collection and make the limits large enough to not clip any of the
object, i.e., for `.Axes.scatter` it would make the limits large enough to not
clip any markers in the scatter.  This is problematic when the object size is
specified in physical space, or figure-relative space, because the transform
from physical units to data limits requires knowing the data limits, and
becomes invalid when the new limits are applied.  This is an inverse
problem that is theoretically solvable (if the object is physically smaller
than the axes), but the extra complexity was not deemed worth it, particularly
as the most common use case is for markers in scatter that are usually small
enough to be accommodated by the default data limit margins.

While the new behavior is algorithmically simpler, it is conditional on
properties of the `.Collection` object:

  1. ``offsets = None``, ``transform`` is a child of ``Axes.transData``: use the paths
     for the automatic limits (i.e. for `.LineCollection` in `.Axes.streamplot`).
  2.  ``offsets != None``, and ``offset_transform`` is child of ``Axes.transData``:

    a) ``transform`` is child of ``Axes.transData``: use the ``path + offset`` for
        limits (i.e., for `.Axes.bar`).
    b) ``transform`` is not a child of ``Axes.transData``: just use the offsets
        for the limits (i.e. for scatter)

  3. otherwise return a null `.Bbox`.

While this seems complicated, the logic is simply to use the information from
the object that are in data space for the limits, but not information that is
in physical units.

log-scale bar() / hist() autolimits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The autolimits computation in `~.Axes.bar` and `~.Axes.hist` when the axes
already uses log-scale has changed to match the computation when the axes is
switched to log-scale after the call to `~.Axes.bar` and `~.Axes.hist`, and
when calling ``bar(..., log=True)`` / ``hist(..., log=True)``: if there are
at least two different bar heights, add the normal axes margins to them (in
log-scale); if there is only a single bar height, expand the axes limits by one
order of magnitude around it and then apply axes margins.


Axes labels spanning multiple rows/columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Axes.label_outer`` now correctly keep the x labels and tick labels visible
for Axes spanning multiple rows, as long as they cover the last row of the Axes
grid.  (This is consistent with keeping the y labels and tick labels visible
for Axes spanning multiple columns as long as they cover the first column of
the Axes grid.)

The ``Axes.is_last_row`` and ``Axes.is_last_col`` methods now correctly return
True for Axes spanning multiple rows, as long as they cover the last row or
column respectively.  Again this is consistent with the behavior for axes
covering the first row or column.

The ``Axes.rowNum`` and ``Axes.colNum`` attributes are deprecated, as they only
refer to the first grid cell covered by the Axes.  Instead, use the new
``ax.get_subplotspec().rowspan`` and ``ax.get_subplotspec().colspan``
properties, which are `range` objects indicating the whole span of rows and
columns covered by the subplot.

(Note that all methods and attributes mentioned here actually only exist on
the ``Subplot`` subclass of `~.axes.Axes`, which is used for grid-positioned Axes but
not for Axes positioned directly in absolute coordinates.)

The `.GridSpec` class gained the ``nrows`` and ``ncols`` properties as more
explicit synonyms for the parameters returned by ``GridSpec.get_geometry``.


Locators
~~~~~~~~
When more than `.Locator.MAXTICKS` ticks are generated, the behavior of
`.Locator.raise_if_exceeds` changed from raising a RuntimeError to emitting a
log at WARNING level.

nonsingular Locators
~~~~~~~~~~~~~~~~~~~~
``Locator.nonsingular`` (introduced in mpl 3.1), ``DateLocator.nonsingular``, and
``AutoDateLocator.nonsingular`` now returns a range ``v0, v1`` with ``v0 <= v1``.
This behavior is consistent with the implementation of ``nonsingular`` by the
``LogLocator`` and ``LogitLocator`` subclasses.

``get_data_ratio``
~~~~~~~~~~~~~~~~~~
``Axes.get_data_ratio`` now takes the axes scale into account (linear, log,
logit, etc.) before computing the y-to-x ratio.  This change allows fixed
aspects to be applied to any combination of x and y scales.

Artist sticky edges
~~~~~~~~~~~~~~~~~~~
Previously, the ``sticky_edges`` attribute of artists was a list of values such
that if an axis limit coincides with a sticky edge, it would not be expanded by
the axes margins (this is the mechanism that e.g. prevents margins from being
added around images).

``sticky_edges`` now have an additional effect on margins application: even if
an axis limit did not coincide with a sticky edge, it cannot *cross* a sticky
edge through margin application -- instead, the margins will only expand the
axis limit until it bumps against the sticky edge.

This change improves the margins of axes displaying a `~.Axes.streamplot`:

- if the streamplot goes all the way to the edges of the vector field, then the
  axis limits are set to match exactly the vector field limits (whereas they
  would sometimes be off by a small floating point error previously).
- if the streamplot does not reach the edges of the vector field (e.g., due to
  the use of ``start_points`` and ``maxlength``), then margins expansion will
  not cross the vector field limits anymore.

This change is also used internally to ensure that polar plots don't display
negative *r* values unless the user really passes in a negative value.

``gid`` in svg output
~~~~~~~~~~~~~~~~~~~~~
Previously, if a figure, axis, legend or some other artists had a custom
``gid`` set (e.g. via ``.set_gid()``), this would not be reflected in
the svg output. Instead a default gid, like ``figure_1`` would be shown.
This is now fixed, such that e.g. ``fig.set_gid("myfigure")`` correctly
shows up as ``<g id="myfigure">`` in the svg file. If you relied on the
gid having the default format, you now need to make sure not to set the
``gid`` parameter of the artists.

Fonts
~~~~~
Font weight guessing now first checks for the presence of the FT_STYLE_BOLD_FLAG
before trying to match substrings in the font name.  In particular, this means
that Times New Roman Bold is now correctly detected as bold, not normal weight.

Color-like checking
~~~~~~~~~~~~~~~~~~~
`matplotlib.colors.is_color_like` used to return True for all string
representations of floats. However, only those with values in 0-1 are valid
colors (representing grayscale values). `.is_color_like` now returns False
for string representations of floats outside 0-1.

Default image interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Images displayed in Matplotlib previously used nearest-neighbor
interpolation, leading to aliasing effects for downscaling and non-integer
upscaling.

New default for :rc:`image.interpolation` is the new option "antialiased".
``imshow(A, interpolation='antialiased')`` will apply a Hanning filter when
resampling the data in A for display (or saving to file) *if* the upsample
rate is less than a factor of three, and not an integer; downsampled data is
always smoothed at resampling.

To get the old behavior, set :rc:`image.interpolation` to the old default "nearest"
(or specify the ``interpolation`` kwarg of `.Axes.imshow`)

To always get the anti-aliasing behavior, no matter what the up/down sample
rate, set :rc:`image.interpolation` to "hanning" (or one of the other filters
available).

Note that the "hanning" filter was chosen because it has only a modest
performance penalty.  Anti-aliasing can be improved with other filters.

rcParams
~~~~~~~~
When using `.RendererSVG` with ``rcParams["svg.image_inline"] ==
True``, externally written images now use a single counter even if the
``renderer.basename`` attribute is overwritten, rather than a counter per
basename.

This change will only affect you if you used ``rcParams["svg.image_inline"] = True``
(the default is False) *and* manually modified ``renderer.basename``.

Changed the default value of :rc:`axes.formatter.limits` from -7, 7 to -5, 6
for better readability.

``add_subplot()``
~~~~~~~~~~~~~~~~~
`.Figure.add_subplot()` and `.pyplot.subplot()` do not accept a *figure*
keyword argument anymore. It only used to work anyway if the passed figure
was ``self`` or the current figure, respectively.

``indicate_inset()``
~~~~~~~~~~~~~~~~~~~~
In <= 3.1.0, `~matplotlib.axes.Axes.indicate_inset` and
`~matplotlib.axes.Axes.indicate_inset_zoom` were documented as returning
a 4-tuple of `~matplotlib.patches.ConnectionPatch`, where in fact they
returned a 4-length list.

They now correctly return a 4-tuple.
`~matplotlib.axes.Axes.indicate_inset` would previously raise an error if
the optional *inset_ax* was not supplied; it now completes successfully,
and returns *None* instead of the tuple of ``ConnectionPatch``.

PGF backend
~~~~~~~~~~~
The pgf backend's get_canvas_width_height now returns the canvas size in
display units rather than in inches, which it previously did.
The new behavior is the correct one given the uses of ``get_canvas_width_height``
in the rest of the codebase.

The pgf backend now includes images using ``\includegraphics`` instead of
``\pgfimage`` if the version of ``graphicx`` is recent enough to support the
``interpolate`` option (this is detected automatically).

`~matplotlib.cbook`
~~~~~~~~~~~~~~~~~~~
The default value of the "obj_type" parameter to ``cbook.warn_deprecated`` has
been changed from "attribute" (a default that was never used internally) to the
empty string.

Testing
~~~~~~~
The test suite no longer turns on the Python fault handler by default.
Set the standard ``PYTHONFAULTHANDLER`` environment variable to do so.

Backend ``supports_blit``
~~~~~~~~~~~~~~~~~~~~~~~~~
Backends do not need to explicitly define the flag ``supports_blit`` anymore.
This is only relevant for backend developers. Backends had to define the flag
``supports_blit``. This is not needed anymore because the blitting capability
is now automatically detected.

Exception changes
~~~~~~~~~~~~~~~~~
Various APIs that raised a `ValueError` for incorrectly typed inputs now raise
`TypeError` instead: `.backend_bases.GraphicsContextBase.set_clip_path`,
``blocking_input.BlockingInput.__call__``, `.cm.register_cmap`, `.dviread.DviFont`,
`.rcsetup.validate_hatch`, ``.rcsetup.validate_animation_writer_path``, `.spines.Spine`,
many classes in the :mod:`matplotlib.transforms` module and :mod:`matplotlib.tri`
package, and Axes methods that take a ``norm`` parameter.

If extra kwargs are passed to `.LogScale`, `TypeError` will now be
raised instead of `ValueError`.

mplot3d auto-registration
~~~~~~~~~~~~~~~~~~~~~~~~~

`mpl_toolkits.mplot3d` is always registered by default now. It is no
longer necessary to import mplot3d to create 3d axes with ::

  ax = fig.add_subplot(111, projection="3d")

`.SymLogNorm` now has a *base* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, `.SymLogNorm` had no *base* keyword argument and the base was
hard-coded to ``base=np.e``. This was inconsistent with the default behavior of
`.SymmetricalLogScale` (which defaults to ``base=10``) and the use of the word
"decade" in the documentation.

In preparation for changing the default base to 10, calling `.SymLogNorm`
without the new *base* keyword argument emits a deprecation warning.
