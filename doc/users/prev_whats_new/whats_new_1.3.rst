.. _whats-new-1-3:

What's new in Matplotlib 1.3 (Aug 01, 2013)
===========================================

.. contents:: Table of Contents
   :depth: 2

.. note::

   matplotlib 1.3 supports Python 2.6, 2.7, 3.2, and 3.3

New in 1.3.1
------------

1.3.1 is a bugfix release, primarily dealing with improved setup and
handling of dependencies, and correcting and enhancing the
documentation.

The following changes were made in 1.3.1 since 1.3.0.

Enhancements
````````````

- Added a context manager for creating multi-page pdfs (see
  `matplotlib.backends.backend_pdf.PdfPages`).

- The WebAgg backend should now have lower latency over heterogeneous
  Internet connections.

Bug fixes
`````````

- Histogram plots now contain the endline.

- Fixes to the Molleweide projection.

- Handling recent fonts from Microsoft and Macintosh-style fonts with
  non-ascii metadata is improved.

- Hatching of fill between plots now works correctly in the PDF
  backend.

- Tight bounding box support now works in the PGF backend.

- Transparent figures now display correctly in the Qt4Agg backend.

- Drawing lines from one subplot to another now works.

- Unit handling on masked arrays has been improved.

Setup and dependencies
``````````````````````

- Now works with any version of pyparsing 1.5.6 or later, without displaying
  hundreds of warnings.

- Now works with 64-bit versions of Ghostscript on MS-Windows.

- When installing from source into an environment without Numpy, Numpy
  will first be downloaded and built and then used to build
  matplotlib.

- Externally installed backends are now always imported using a
  fully-qualified path to the module.

- Works with newer version of wxPython.

- Can now build with a PyCXX installed globally on the system from source.

- Better detection of Gtk3 dependencies.

Testing
```````

- Tests should now work in non-English locales.

- PEP8 conformance tests now report on locations of issues.


New plotting features
---------------------

`~.xkcd`-style sketch plotting
``````````````````````````````
To give your plots a sense of authority that they may be missing,
Michael Droettboom (inspired by the work of many others in
:ghpull:`1329`) has added an `xkcd-style <https://xkcd.com/>`__ sketch
plotting mode.  To use it, simply call `matplotlib.pyplot.xkcd`
before creating your plot. For really fine control, it is also possible
to modify each artist's sketch parameters individually with
:meth:`matplotlib.artist.Artist.set_sketch_params`.

.. figure:: ../../gallery/showcase/images/sphx_glr_xkcd_001.png
   :target: ../../gallery/showcase/xkcd.html
   :align: center
   :scale: 50

Updated Axes3D.contour methods
------------------------------
Damon McDougall updated the
:meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.tricontour` and
:meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.tricontourf` methods to allow 3D
contour plots on abitrary unstructured user-specified triangulations.

.. figure:: ../../gallery/mplot3d/images/sphx_glr_tricontour3d_001.png
   :target: ../../gallery/mplot3d/tricontour3d.html
   :align: center
   :scale: 50

New eventplot plot type
```````````````````````
Todd Jennings added a :func:`~matplotlib.pyplot.eventplot` function to
create multiple rows or columns of identical line segments

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_eventplot_demo_001.png
   :target: ../../gallery/lines_bars_and_markers/eventplot_demo.html
   :align: center
   :scale: 50

As part of this feature, there is a new
:class:`~matplotlib.collections.EventCollection` class that allows for
plotting and manipulating rows or columns of identical line segments.

Triangular grid interpolation
`````````````````````````````
Geoffroy Billotey and Ian Thomas added classes to perform
interpolation within triangular grids:
(:class:`~matplotlib.tri.LinearTriInterpolator` and
:class:`~matplotlib.tri.CubicTriInterpolator`) and a utility class to
find the triangles in which points lie
(:class:`~matplotlib.tri.TrapezoidMapTriFinder`).  A helper class to
perform mesh refinement and smooth contouring was also added
(:class:`~matplotlib.tri.UniformTriRefiner`).  Finally, a class
implementing some basic tools for triangular mesh improvement was
added (:class:`~matplotlib.tri.TriAnalyzer`).

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_tricontour_smooth_user_001.png
   :target: ../../gallery/images_contours_and_fields/tricontour_smooth_user.html
   :align: center
   :scale: 50

Baselines for stackplot
```````````````````````
Till Stensitzki added non-zero baselines to
:func:`~matplotlib.pyplot.stackplot`.  They may be symmetric or
weighted.

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_stackplot_demo_001.png
   :target: ../../gallery/lines_bars_and_markers/stackplot_demo.html
   :align: center
   :scale: 50

Rectangular colorbar extensions
```````````````````````````````
Andrew Dawson added a new keyword argument *extendrect* to
:meth:`~matplotlib.pyplot.colorbar` to optionally make colorbar
extensions rectangular instead of triangular.

More robust boxplots
````````````````````
Paul Hobson provided a fix to the :func:`~matplotlib.pyplot.boxplot`
method that prevent whiskers from being drawn inside the box for
oddly distributed data sets.

Calling subplot() without arguments
```````````````````````````````````
A call to :func:`~matplotlib.pyplot.subplot` without any arguments now
acts the same as ``subplot(111)`` or ``subplot(1, 1, 1)`` -- it creates one
axes for the whole figure. This was already the behavior for both
:func:`~matplotlib.pyplot.axes` and
:func:`~matplotlib.pyplot.subplots`, and now this consistency is
shared with :func:`~matplotlib.pyplot.subplot`.

Drawing
-------

Independent alpha values for face and edge colors
`````````````````````````````````````````````````
Wes Campaigne modified how :class:`~matplotlib.patches.Patch` objects are
drawn such that (for backends supporting transparency) you can set different
alpha values for faces and edges, by specifying their colors in RGBA format.
Note that if you set the alpha attribute for the patch object (e.g. using
:meth:`~matplotlib.patches.Patch.set_alpha` or the ``alpha`` keyword
argument), that value will override the alpha components set in both the
face and edge colors.

Path effects on lines
`````````````````````
Thanks to Jae-Joon Lee, path effects now also work on plot lines.

.. figure:: ../../gallery/misc/images/sphx_glr_patheffect_demo_001.png
   :target: ../../gallery/misc/patheffect_demo.html
   :align: center
   :scale: 50

Easier creation of colormap and normalizer for levels with colors
`````````````````````````````````````````````````````````````````
Phil Elson added the :func:`matplotlib.colors.from_levels_and_colors`
function to easily create a colormap and normalizer for representation
of discrete colors for plot types such as
:func:`matplotlib.pyplot.pcolormesh`, with a similar interface to that
of `matplotlib.pyplot.contourf`.

Full control of the background color
````````````````````````````````````
Wes Campaigne and Phil Elson fixed the Agg backend such that PNGs are
now saved with the correct background color when
``fig.patch.get_alpha()`` is not 1.

Improved ``bbox_inches="tight"`` functionality
``````````````````````````````````````````````
Passing ``bbox_inches="tight"`` through to `.pyplot.savefig` now takes
into account *all* artists on a figure - this was previously not the
case and led to several corner cases which did not function as
expected.

Initialize a rotated rectangle
``````````````````````````````
Damon McDougall extended the :class:`~matplotlib.patches.Rectangle`
constructor to accept an *angle* kwarg, specifying the rotation of a
rectangle in degrees.

Text
----

Anchored text support
`````````````````````
The SVG and pgf backends are now able to save text alignment
information to their output formats. This allows to edit text elements
in saved figures, using Inkscape for example, while preserving their
intended position. For SVG please note that you'll have to disable
the default text-to-path conversion (``mpl.rc('svg',
fonttype='none')``).

Better vertical text alignment and multi-line text
``````````````````````````````````````````````````
The vertical alignment of text is now consistent across backends.  You
may see small differences in text placement, particularly with rotated
text.

If you are using a custom backend, note that the `~.RendererBase.draw_text` renderer
method is now passed the location of the baseline, not the location of
the bottom of the text bounding box.

Multi-line text will now leave enough room for the height of very tall
or very low text, such as superscripts and subscripts.

Left and right side axes titles
```````````````````````````````
Andrew Dawson added the ability to add axes titles flush with the left
and right sides of the top of the axes using a new keyword argument
*loc* to :func:`~matplotlib.pyplot.title`.

Improved manual contour plot label positioning
``````````````````````````````````````````````
Brian Mattern modified the manual contour plot label positioning code
to interpolate along line segments and find the actual closest point
on a contour to the requested position. Previously, the closest path
vertex was used, which, in the case of straight contours was sometimes
quite distant from the requested location. Much more precise label
positioning is now possible.

Configuration (rcParams)
------------------------

Quickly find rcParams
`````````````````````
Phil Elson made it easier to search for rcParameters by passing a
valid regular expression to :func:`matplotlib.RcParams.find_all`.
:class:`matplotlib.RcParams` now also has a pretty repr and str
representation so that search results are printed prettily:

    >>> import matplotlib
    >>> print(matplotlib.rcParams.find_all('\.size'))
    RcParams({'font.size': 12,
              'xtick.major.size': 4,
              'xtick.minor.size': 2,
              'ytick.major.size': 4,
              'ytick.minor.size': 2})

``axes.xmargin`` and ``axes.ymargin`` added to rcParams
```````````````````````````````````````````````````````
:rc:`axes.xmargin` and :rc:`axes.ymargin` were added
to configure the default margins used.  Previously they were
hard-coded to default to 0, default value of both rcParam values is 0.

Changes to font rcParams
````````````````````````
The ``font.*`` rcParams now affect only text objects created after the
rcParam has been set, and will not retroactively affect already
existing text objects.  This brings their behavior in line with most
other rcParams.

Added :rc:`savefig.jpeg_quality`
````````````````````````````````
rcParam value :rc:`savefig.jpeg_quality` was added so that the user can
configure the default quality used when a figure is written as a JPEG.
The default quality is 95; previously, the default quality was 75.
This change minimizes the artifacting inherent in JPEG images,
particularly with images that have sharp changes in color as plots
often do.

Backends
--------

WebAgg backend
``````````````
Michael Droettboom, Phil Elson and others have developed a new
backend, WebAgg, to display figures in a web browser.  It works with
animations as well as being fully interactive.

.. image:: /_static/webagg_screenshot.png

Future versions of matplotlib will integrate this backend with the
IPython notebook for a fully web browser based plotting frontend.

Remember save directory
```````````````````````
Martin Spacek made the save figure dialog remember the last directory
saved to. The default is configurable with the new :rc:`savefig.directory`
rcParam in :file:`matplotlibrc`.

Documentation and examples
--------------------------

Numpydoc docstrings
```````````````````
Nelle Varoquaux has started an ongoing project to convert matplotlib's
docstrings to numpydoc format.  See `MEP10
<https://github.com/matplotlib/matplotlib/wiki/Mep10>`__ for more
information.

Example reorganization
``````````````````````
Tony Yu has begun work reorganizing the examples into more meaningful
categories.  The new gallery page is the fruit of this ongoing work.
See `MEP12 <https://github.com/matplotlib/matplotlib/wiki/MEP12>`__ for
more information.

Examples now use subplots()
```````````````````````````
For the sake of brevity and clarity, most of the :ref:`examples
<examples-index>` now use the newer
:func:`~matplotlib.pyplot.subplots`, which creates a figure and one
(or multiple) axes object(s) in one call. The old way involved a call
to :func:`~matplotlib.pyplot.figure`, followed by one (or multiple)
:func:`~matplotlib.pyplot.subplot` calls.

Infrastructure
--------------

Housecleaning
`````````````
A number of features that were deprecated in 1.2 or earlier, or have
not been in a working state for a long time have been removed.
Highlights include removing the Qt version 3 backends, and the FltkAgg
and Emf backends.  See :ref:`changes_in_1_3` for a complete list.

New setup script
````````````````
matplotlib 1.3 includes an entirely rewritten setup script.  We now
ship fewer dependencies with the tarballs and installers themselves.
Notably, pytz_, dateutil_, pyparsing_ and six_ are no longer
included with matplotlib.  You can either install them manually first,
or let pip_ install them as dependencies along with matplotlib.  It
is now possible to not include certain subcomponents, such as the unit
test data, in the install.  See :file:`setup.cfg.template` for more
information.

.. _dateutil: https://pypi.org/project/python-dateutil/
.. _pip: https://pypi.org/project/pip/
.. _pyparsing: https://pypi.org/project/pyparsing/
.. _pytz: https://pypi.org/project/pytz/
.. _six: https://pypi.org/project/six/

XDG base directory support
``````````````````````````
On Linux, matplotlib now uses the `XDG base directory specification
<https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_ to
find the :file:`matplotlibrc` configuration file. :file:`matplotlibrc` should
now be kept in :file:`~/.config/matplotlib`, rather than :file:`~/.matplotlib`.
If your configuration is found in the old location, it will still be used, but
a warning will be displayed.

Catch opening too many figures using pyplot
```````````````````````````````````````````
Figures created through `.pyplot.figure` are retained until they are
explicitly closed.  It is therefore common for new users of matplotlib
to run out of memory when creating a large series of figures in a loop
without closing them.

matplotlib will now display a `RuntimeWarning` when too many figures
have been opened at once.  By default, this is displayed for 20 or
more figures, but the exact number may be controlled using the
``figure.max_open_warning`` rcParam.
