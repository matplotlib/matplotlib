.. _whats-new:

==========================
 What's new in matplotlib
==========================

For a list of all of the issues and pull requests since the last
revision, see the :ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 3


..
   For a release, add a new section after this, then comment out the include
   and toctree below by indenting them. Uncomment them after the release.
.. include:: next_whats_new/README.rst
.. toctree::
   :glob:
   :maxdepth: 1

   next_whats_new/*

New in Matplotlib 2.1
=====================

New TransformedPatchPath caching object
---------------------------------------

A newly added :class:`~matplotlib.transforms.TransformedPatchPath` provides a
means to transform a :class:`~matplotlib.patches.Patch` into a
:class:`~matplotlib.path.Path` via a :class:`~matplotlib.transforms.Transform`
while caching the resulting path. If neither the patch nor the transform have
changed, a cached copy of the path is returned.

This class differs from the older
:class:`~matplotlib.transforms.TransformedPath` in that it is able to refresh
itself based on the underlying patch while the older class uses an immutable
path.


CheckButtons widget get_status function
---------------------------------------

A :func:`get_status` function has been added the :class:`matplotlib.widgets.CheckButtons` class. This :func:`get_status` function allows user to query the status (True/False) of all of the buttons in the CheckButtons object.


Abstract base class for movie writers
-------------------------------------

The new :class:`~matplotlib.animation.AbstractMovieWriter` class defines
the API required by a class that is to be used as the `writer` in the
`save` method of the :class:`~matplotlib.animation.Animation` class.
The existing :class:`~matplotlib.animation.MovieWriter` class now derives
from the new abstract base class.


Add fill_bar argument to ``AnchoredSizeBar``
--------------------------------------------

The mpl_toolkits class
:class:`~mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar` now has an
additional ``fill_bar`` argument, which makes the size bar a solid rectangle
instead of just drawing the border of the rectangle. The default is ``None``,
and whether or not the bar will be filled by default depends on the value of
``size_vertical``. If ``size_vertical`` is nonzero, ``fill_bar`` will be set to
``True``. If ``size_vertical`` is zero then ``fill_bar`` will be set to
``False``. If you wish to override this default behavior, set ``fill_bar`` to
``True`` or ``False`` to unconditionally always or never use a filled patch
rectangle for the size bar.


Annotation can use a default arrow style
----------------------------------------

Annotations now use the default arrow style when setting `arrowprops={}`,
rather than no arrow (the new behavior actually matches the documentation).


Orthographic projection for mplot3d
-----------------------------------
:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` now accepts ``proj_type`` kwarg and has a method :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.set_proj_type`. The default option is ``'persp'`` as before, and supplying ``'ortho'`` enables orthographic view.


``voxels`` function for mplot3d
-------------------------------
:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` now has a ``voxels`` method, for
visualizing boolean 3d data. Uses could include plotting a sparse 3D heat map,
or visualizing a volumetric model.


Barbs and Quiver Support Dates
------------------------------

When using the :func:`quiver` and :func:`barbs` plotting methods,
it is now possible to pass dates, just like for other methods like
:func:`plot`. This also allows these functions to handle values
that need unit-conversion applied.


Hexbin default line color
-------------------------

The default ``linecolor`` kwarg for :func:`hexbin` is now ``'face'``, and
supplying ``'none'`` now prevents lines from being drawn around the hexagons.


figure.legend() can be called without arguments
-----------------------------------------------

Calling :func:`figure.legend` can now be done with no arguments. In this case a
legend will be created that contains all the artists on all the axes contained
within the figure.


New parameter `clear` for :func:`~matplotlib.pyplot.figure`
-----------------------------------------------------------

When the pyplot's function :func:`~matplotlib.pyplot.figure` is called
with a ``num`` parameter, a new window is only created if no existing
window with the same value exists. A new bool parameter `clear` was
added for explicitly clearing its existing contents. This is particularly
useful when utilized in interactive sessions. Since
:func:`~matplotlib.pyplot.subplots` also accepts keyword arguments
from :func:`~matplotlib.pyplot.figure`, it can also be used there::

   import matplotlib.pyplot as plt

   fig0 = plt.figure(num=1)
   fig0.suptitle("A fancy plot")
   print("fig0.texts: ", [t.get_text() for t in fig0.texts])

   fig1 = plt.figure(num=1, clear=False)  # do not clear contents of window
   fig1.text(0.5, 0.5, "Really fancy!")
   print("fig0 is fig1: ",  fig0 is fig1)
   print("fig1.texts: ", [t.get_text() for t in fig1.texts])

   fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)  # clear contents
   print("fig0 is fig2: ",  fig0 is fig2)
   print("fig2.texts: ", [t.get_text() for t in fig2.texts])

   # The output:
   # fig0.texts:  ['A fancy plot']
   # fig0 is fig1:  True
   # fig1.texts:  ['A fancy plot', 'Really fancy!']
   # fig0 is fig2:  True
   # fig2.texts:  []

AVConv writer is back
---------------------
Correct a bug that prevented detection of AVconv for matplotlib.animation.


Invalid (Non-finite) Axis Limit Error
-------------------------------------

When using :func:`set_xlim` and :func:`set_ylim`, passing non-finite values now
results in a ValueError. The previous behavior resulted in the limits being
erroneously reset to `(-0.001, 0.001)`.


Metadata savefig kwarg
----------------------

:func:`~matplotlib.pyplot.savefig` now accepts `metadata` as a keyword argument.
It can be used to store key/value pairs in the image metadata.

Supported formats and backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* 'png' with Agg backend
* 'pdf' with PDF backend (see
  :func:`~matplotlib.backends.backend_pdf.PdfFile.writeInfoDict` for a list of
  supported keywords)
* 'eps' and 'ps' with PS backend (only 'Creator' key is accepted)

Example
~~~~~~~
::

    plt.savefig('test.png', metadata={'Software': 'My awesome software'})



Specify minimum value to format as scalar for ``LogFormatterMathtext``
----------------------------------------------------------------------

``LogFormatterMathtext`` now includes the option to specify a minimum value
exponent to format as a scalar (ie. 0.001 instead of 10^-3).


Multiple legend keys for legend entries
---------------------------------------

A legend entry can now contain more than one legend key. The extended
``HandlerTuple`` class now accepts two parameters: ``ndivide`` divides the
legend area in the specified number of sections; ``pad`` changes the padding
between the legend keys.

.. figure:: /gallery/text_labels_and_annotations/images/sphx_glr_legend_demo_004.png
   :target: ../../gallery/text_labels_and_annotations/legend_demo.html
   :align: center
   :scale: 50

   Multiple Legend Keys


Path simplification updates
---------------------------

Line simplification controlled by the ``path.simplify`` and
``path.simplify_threshold`` parameters has been improved. You should
notice better rendering performance when plotting large amounts of
data (as long as the above parameters are set accordingly). Only the
line segment portion of paths will be simplified -- if you are also
drawing markers and experiencing problems with rendering speed, you
should consider using the ``markevery`` option to ``plot``.
See the :ref:`performance` section in the usage tutorial for more
information.

The simplification works by iteratively merging line segments
into a single vector until the next line segment's perpendicular
distance to the vector (measured in display-coordinate space)
is greater than the ``path.simplify_threshold`` parameter. Thus, higher
values of ``path.simplify_threshold`` result in quicker rendering times.
If you are plotting just to explore data and not for publication quality,
pixel perfect plots, then a value of ``1.0`` can be safely used. If you
want to make sure your plot reflects your data *exactly*, then you should
set ``path.simplify`` to false and/or ``path.simplify_threshold`` to ``0``.
Matplotlib currently defaults to a conservative value of ``1/9``, smaller
values are unlikely to cause any visible differences in your plots.


Added `matplotlib.ticker.PercentFormatter`
------------------------------------------

The new formatter has some nice features like being able to convert from
arbitrary data scales to percents, a customizable percent symbol and
either automatic or manual control over the decimal points.


New quiverkey angle kwarg
-------------------------

Plotting a :func:`quiverkey`  now admits the ``angle`` kwarg,
which sets the angle at which to draw the key arrow.


Reproducible PS, PDF and SVG output
-----------------------------------

The ``SOURCE_DATE_EPOCH`` environment variable can now be used to set
the timestamp value in the PS and PDF outputs. See
https://reproducible-builds.org/specs/source-date-epoch/

Alternatively, calling ``savefig`` with ``metadata={'creationDate': None}``
will omit the timestamp altogether.

The reproducibility of the output from the PS and PDF backends has so
far been tested using various plot elements but only default values of
options such as ``{ps,pdf}.fonttype`` that can affect the output at a
low level, and not with the mathtext or usetex features. When
matplotlib calls external tools (such as PS distillers or LaTeX) their
versions need to be kept constant for reproducibility, and they may
add sources of nondeterminism outside the control of matplotlib.

For SVG output, the ``svg.hashsalt`` rc parameter has been added in an
earlier release. This parameter changes some random identifiers in the
SVG file to be deterministic. The downside of this setting is that if
more than one file is generated using deterministic identifiers
and they end up as parts of one larger document, the identifiers can
collide and cause the different parts to affect each other.

These features are now enabled in the tests for the PDF and SVG
backends, so most test output files (but not all of them) are now
deterministic.


Colormap reversed method
------------------------

The methods :meth:`~matplotlib.colors.LinearSegmentedColormap.reversed` and
:meth:`~matplotlib.colors.ListedColormap.reversed` return a reversed
instance of the Colormap. This implements a way for any Colormap to be
reversed.

`Collection` offsets are no longer implicitly flattened
-------------------------------------------------------

`Collection` (and thus `scatter` -- both 2D and 3D) no longer implicitly
flattens its offsets.  As a consequence, `scatter`'s x and y arguments can no
longer be 2+-dimensional arrays.


`Artist.setp` (and `pyplot.setp`) accept a `file` argument
----------------------------------------------------------

The argument is keyword-only. It allows an output file other than
`sys.stdout` to be specified. It works exactly like the `file` argument
to `print`.



Maximum streamline length and integration direction can now be specified
------------------------------------------------------------------------

This allows to follow the vector field for a longer time and can enhance the
visibility of the flow pattern in some use cases.


`Axis.set_tick_params` now responds to 'rotation'
-------------------------------------------------

Bulk setting of tick label rotation is now possible via :func:`set_tick_params` using the `rotation` keyword.

Example
~~~~~~~
::

    ax.xaxis.set_tick_params(which='both', rotation=90)

Users can now toggle shading in 3D bar plots
--------------------------------------------

A new ``shade`` parameter has been added the 3D bar plotting method.
The default behavior remains to shade the bars, but now users
have the option of setting ``shade`` to ``False``.


Example
~~~~~~~
::

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(7,3))
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.arange(2)
    y = np.arange(3)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d
    ax1.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    fig.canvas.draw()


New which Parameter for autofmt_xdate
-------------------------------------

A ``which`` parameter now exists for the method :func:`autofmt_xdate`. This
allows a user to format ``major``, ``minor`` or ``both`` tick labels
selectively. If ``which`` is ``None`` (default) then the method will rotate
``major`` tick labels.

Example
~~~~~~~
::

    autofmt_xdate(self, bottom=0.2, rotation=30, ha='right', which='minor')


New Figure Parameter for subplot2grid
-------------------------------------

A ``fig`` parameter now exists for the method :func:`subplot2grid`.  This allows
a user to specify the figure where the subplots will be created.  If ``fig``
is ``None`` (default) then the method will use the current figure retrieved by
:func:`gcf`.

Example
~~~~~~~
::

    subplot2grid(shape, loc, rowspan=1, colspan=1, fig=myfig)


Interpolation in fill_betweenx
------------------------------

The ``interpolate`` parameter now exists for the method :func:`fill_betweenx`.
This allows a user to interpolate the data and fill the areas in the crossover
points, similarly to :func:`fill_between`.


Validation of line style rcParams
---------------------------------

Stricter validation
~~~~~~~~~~~~~~~~~~~
The validation of rcParams that are related to line styles
(``lines.linestyle``, ``boxplot.*.linestyle``, ``grid.linestyle`` and
``contour.negative_linestyle``) now effectively checks that the values
are valid line styles. Strings like ``dashed`` or ``--`` are accepted,
as well as even-length sequences of on-off ink like ``[1, 1.65]``. In
this latter case, the offset value is handled internally and should *not*
be provided by the user.

The validation is case-insensitive.

Deprecation of the former validators for ``contour.negative_linestyle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The new validation scheme replaces the former one used for the
``contour.negative_linestyle`` rcParams, that was limited to ``solid``
and ``dashed`` line styles.

The former public validation functions ``validate_negative_linestyle``
and ``validate_negative_linestyle_legacy`` will be deprecated in 2.1 and
may be removed in 2.3. There are no public functions to replace them.

Examples of use
~~~~~~~~~~~~~~~
::

    grid.linestyle             : (1, 3)   # loosely dotted grid lines
    contour.negative_linestyle : dashdot  # previously only solid or dashed


New in matplotlib 2.0
=====================

.. note::

   matplotlib 2.0 supports Python 2.7, and 3.4+



Default style changes
---------------------

The major changes in v2.0 are related to overhauling the default styles.

.. toctree::
   :maxdepth: 2

   dflt_style_changes


Improved color conversion API and RGBA support
----------------------------------------------

The :mod:`~matplotlib.colors` gained a new color conversion API with
full support for the alpha channel.  The main public functions are
:func:`~matplotlib.colors.is_color_like`, :func:`matplotlib.colors.to_rgba`,
:func:`matplotlib.colors.to_rgba_array` and :func:`~matplotlib.colors.to_hex`.
RGBA quadruplets are encoded in hex format as `#rrggbbaa`.

A side benefit is that the Qt options editor now allows setting the alpha
channel of the artists as well.


New Configuration (rcParams)
----------------------------

New rcparams added

+---------------------------------+--------------------------------------------------+
| Parameter                       | Description                                      |
+=================================+==================================================+
|`date.autoformatter.year`        | format string for 'year' scale dates             |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.month`       | format string for 'month' scale dates            |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.day`         | format string for 'day' scale dates              |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.hour`        | format string for 'hour' scale times             |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.minute`      | format string for 'minute' scale times           |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.second`      | format string for 'second' scale times           |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.microsecond` | format string for 'microsecond' scale times      |
+---------------------------------+--------------------------------------------------+
|`scatter.marker`                 | default marker for scatter plot                  |
+---------------------------------+--------------------------------------------------+
|`svg.hashsalt`                   | see note                                         |
+---------------------------------+--------------------------------------------------+
|`xtick.top`, `xtick.minor.top`,  | Control where major and minor ticks are drawn.   |
|`xtick.major.top`                | The global values are `and` ed with the          |
|`xtick.bottom`,                  | corresponding major/minor values.                |
|`xtick.minor.bottom`,            |                                                  |
|`xtick.major.bottom`             |                                                  |
|`ytick.left`, `ytick.minor.left`,|                                                  |
|`ytick.major.left`               |                                                  |
|`ytick.right`,                   |                                                  |
|`ytick.minor.right`,             |                                                  |
|`ytick.major.right`              |                                                  |
+---------------------------------+--------------------------------------------------+
|`hist.bins`                      | The default number of bins to use in             |
|                                 | `~matplotlib.axes.Axes.hist`.  This can be an    |
|                                 | `int`, a list of floats, or ``'auto'`` if numpy  |
|                                 | >= 1.11 is installed.                            |
+---------------------------------+--------------------------------------------------+
|`lines.scale_dashes`             | Whether the line dash patterns should scale with |
|                                 | linewidth.                                       |
+---------------------------------+--------------------------------------------------+
|`axes.formatter.offset_threshold`| Minimum number of digits saved in tick labels    |
|                                 | that triggers using an offset.                   |
+---------------------------------+--------------------------------------------------+



Added ``svg.hashsalt`` key to rcParams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``svg.hashsalt`` is ``None`` (which it is by default), the svg
backend uses ``uuid4`` to generate the hash salt.  If it is not
``None``, it must be a string that is used as the hash salt instead of
``uuid4``.  This allows for deterministic SVG output.


Removed the ``svg.image_noscale`` rcParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a result of the extensive changes to image handling, the
``svg.image_noscale`` rcParam has been removed.  The same
functionality may be achieved by setting ``interpolation='none'`` on
individual images or globally using the ``image.interpolation``
rcParam.


Qualitative colormaps
---------------------

ColorBrewer's "qualitative" colormaps ("Accent", "Dark2", "Paired",
"Pastel1", "Pastel2", "Set1", "Set2", "Set3") were intended for discrete
categorical data, with no implication of value, and therefore have been
converted to ``ListedColormap`` instead of ``LinearSegmentedColormap``, so
the colors will no longer be interpolated and they can be used for
choropleths, labeled image features, etc.



Axis offset label now responds to `labelcolor`
----------------------------------------------

Axis offset labels are now colored the same as axis tick markers when `labelcolor` is altered.

Improved offset text choice
---------------------------
The default offset-text choice was changed to only use significant digits that
are common to all ticks (e.g. 1231..1239 -> 1230, instead of 1231), except when
they straddle a relatively large multiple of a power of ten, in which case that
multiple is chosen (e.g. 1999..2001->2000).


Style parameter blacklist
-------------------------

In order to prevent unexpected consequences from using a style, style
files are no longer able to set parameters that affect things
unrelated to style.  These parameters include::

  'interactive', 'backend', 'backend.qt4', 'webagg.port',
  'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
  'toolbar', 'timezone', 'datapath', 'figure.max_open_warning',
  'savefig.directory', 'tk.window_focus', 'docstring.hardcopy'


Change in default font
----------------------

The default font used by matplotlib in text has been changed to DejaVu Sans and
DejaVu Serif for the sans-serif and serif families, respectively. The DejaVu
font family is based on the previous matplotlib default --Bitstream Vera-- but
includes a much wider range of characters.

The default mathtext font has been changed from Computer Modern to the DejaVu
family to maintain consistency with regular text. Two new options for the
``mathtext.fontset`` configuration parameter have been added: ``dejavusans``
(default) and ``dejavuserif``. Both of these options use DejaVu glyphs whenever
possible and fall back to STIX symbols when a glyph is not found in DejaVu. To
return to the previous behavior, set the rcParam ``mathtext.fontset`` to ``cm``.


Faster text rendering
---------------------

Rendering text in the Agg backend is now less fuzzy and about 20%
faster to draw.


Improvements for the Qt figure options editor
---------------------------------------------

Various usability improvements were implemented for the Qt figure options
editor, among which:

- Line style entries are now sorted without duplicates.
- The colormap and normalization limits can now be set for images.
- Line edits for floating values now display only as many digits as necessary
  to avoid precision loss.  An important bug was also fixed regarding input
  validation using Qt5 and a locale where the decimal separator is ",".
- The axes selector now uses shorter, more user-friendly names for axes, and
  does not crash if there are no axes.
- Line and image entries using the default labels ("_lineX", "_imageX") are now
  sorted numerically even when there are more than 10 entries.


Improved image support
----------------------

Prior to version 2.0, matplotlib resampled images by first applying
the color map and then resizing the result.  Since the resampling was
performed on the colored image, this introduced colors in the output
image that didn't actually exist in the color map.  Now, images are
resampled first (and entirely in floating-point, if the input image is
floating-point), and then the color map is applied.

In order to make this important change, the image handling code was
almost entirely rewritten.  As a side effect, image resampling uses
less memory and fewer datatype conversions than before.

The experimental private feature where one could "skew" an image by
setting the private member ``_image_skew_coordinate`` has been
removed.  Instead, images will obey the transform of the axes on which
they are drawn.

Non-linear scales on image plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`imshow` now draws data at the requested points in data space after the
application of non-linear scales.

The image on the left demonstrates the new, correct behavior.
The old behavior can be recreated using :func:`pcolormesh` as
demonstrated on the right.


.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.arange(30).reshape(5, 6)
    x = np.linspace(0, 6, 7)
    y = 10**np.linspace(0, 5, 6)
    X, Y = np.meshgrid(x, y)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    ax1.imshow(data, aspect="auto", extent=(0, 6, 1e0, 1e5), interpolation='nearest')
    ax1.set_yscale('log')
    ax1.set_title('Using ax.imshow')

    ax2.pcolormesh(x, y, np.flipud(data))
    ax2.set_yscale('log')
    ax2.set_title('Using ax.pcolormesh')
    ax2.autoscale('tight')

    plt.show()


This can be understood by analogy to plotting a histogram with linearly spaced bins
with a logarithmic x-axis.  Equal sized bins will be displayed as wider for small
*x* and narrower for large *x*.



Support for HiDPI (Retina) displays in the NbAgg and WebAgg backends
--------------------------------------------------------------------

The NbAgg and WebAgg backends will now use the full resolution of your
high-pixel-density display.

Change in the default animation codec
-------------------------------------

The default animation codec has been changed from ``mpeg4`` to ``h264``,
which is more efficient. It can be set via the ``animation.codec`` rcParam.

Deprecated support for mencoder in animation
--------------------------------------------

The use of mencoder for writing video files with mpl is problematic;
switching to ffmpeg is strongly advised.  All support for mencoder
will be removed in version 2.2.

Boxplot Zorder Keyword Argument
-------------------------------

The ``zorder`` parameter now exists for :func:`boxplot`. This allows the zorder
of a boxplot to be set in the plotting function call.

::

    boxplot(np.arange(10), zorder=10)

Filled ``+`` and ``x`` markers
------------------------------

New fillable *plus* and *x* markers have been added. See
the :mod:`~matplotlib.markers` module and
:ref:`marker reference <sphx_glr_gallery_lines_bars_and_markers_marker_reference.py>`
examples.

`rcount` and `ccount` for `plot_surface()`
------------------------------------------

As of v2.0, mplot3d's :func:`~mpl_toolkits.mplot3d.axes3d.plot_surface` now
accepts `rcount` and `ccount` arguments for controlling the sampling of the
input data for plotting. These arguments specify the maximum number of
evenly spaced samples to take from the input data. These arguments are
also the new default sampling method for the function, and is
considered a style change.

The old `rstride` and `cstride` arguments, which specified the size of the
evenly spaced samples, become the default when 'classic' mode is invoked,
and are still available for use. There are no plans for deprecating these
arguments.

Streamplot Zorder Keyword Argument Changes
------------------------------------------

The ``zorder`` parameter for :func:`streamplot` now has default
value of ``None`` instead of ``2``. If ``None`` is given as ``zorder``,
:func:`streamplot` has a default ``zorder`` of
``matplotlib.lines.Line2D.zorder``.

.. _gc_get_hatch_color_wn:

Extension to `matplotlib.backend_bases.GraphicsContextBase`
-----------------------------------------------------------

To support standardizing hatch behavior across the backends we ship
the `matplotlib.backend_bases.GraphicsContextBase.get_hatch_color`
method as added to `matplotlib.backend_bases.GraphicsContextBase`.
This is only used during the render process in the backends we ship so
will not break any third-party backends.

If you maintain a third-party backend which extends
`~matplotlib.backend_bases.GraphicsContextBase` this method is now
available to you and should be used to color hatch patterns.

Previous Whats New
==================

.. toctree::
   :glob:
   :maxdepth: 1

   prev_whats_new/whats_new_*
   prev_whats_new/changelog
