.. _whats-new:

************************
What's new in matplotlib
************************

This page just covers the highlights -- for the full story, see the
`CHANGELOG <http://matplotlib.sourceforge.net/_static/CHANGELOG>`_

.. note::
   Matplotlib version 1.1 is the last major release compatible with Python
   versions 2.4 to 2.7.  The next major release will support
   versions 2.6, 2.7, and 3.1 and higher.

.. _whats-new-1-1:

new in matplotlib-1.1
=====================

Sankey Diagrams
---------------

Kevin Davies has extended Yannick Copin's original Sankey example into a module
(:mod:`~matplotlib.sankey`) and provided new examples
(:ref:`api-sankey_demo_basics`, :ref:`api-sankey_demo_links`,
:ref:`api-sankey_demo_rankine`).

.. plot:: mpl_examples/api/sankey_demo_rankine.py

Animation
---------

Ryan May has written a backend-independent framework for creating
animated figures. The :mod:`~matplotlib.animation` module is intended
to replace the backend-specific examples formerly in the
:ref:`examples-index` listings.  Examples using the new framework are
in :ref:`animation-examples-index`; see the entrancing :ref:`double
pendulum <animation-double_pendulum_animated>` which uses
:meth:`matplotlib.animation.Animation.save` to create the movie below.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/32cjc6V0OZY" frameborder="0" allowfullscreen></iframe>

This should be considered as a beta release of the framework;
please try it and provide feedback.


Tight Layout
------------

A frequent issue raised by users of matplotlib is the lack of a layout
engine to nicely space out elements of the plots. While matplotlib still
adheres to the philosphy of giving users complete control over the placement
of plot elements, Jae-Joon Lee created the :mod:`~matplotlib.tight_layout`
module and introduced a new
command :func:`~matplotlib.pyplot.tight_layout`
to address the most common layout issues.

.. plot::

    plt.rcParams['savefig.facecolor'] = "0.8"
    plt.rcParams['figure.figsize'] = 4, 3

    fig, axes_list = plt.subplots(2, 1)
    for ax in axes_list.flat:
        ax.set(xlabel="x-label", ylabel="y-label", title="before tight_layout")
	ax.locator_params(nbins=3)

    plt.show()

    plt.rcParams['savefig.facecolor'] = "0.8"
    plt.rcParams['figure.figsize'] = 4, 3

    fig, axes_list = plt.subplots(2, 1)
    for ax in axes_list.flat:
        ax.set(xlabel="x-label", ylabel="y-label", title="after tight_layout")
	ax.locator_params(nbins=3)

    plt.tight_layout()
    plt.show()

The usage of this functionality can be as simple as ::

    plt.tight_layout()

and it will adjust the spacing between subplots
so that the axis labels do not overlap with neighboring subplots. A
:ref:`plotting-guide-tight-layout` has been created to show how to use
this new tool.

PyQT4, PySide, and IPython
--------------------------

Gerald Storer made the Qt4 backend compatible with PySide as
well as PyQT4.  At present, however, PySide does not support
the PyOS_InputHook mechanism for handling gui events while
waiting for text input, so it cannot be used with the new
version 0.11 of `IPython <http://ipython.org>`_. Until this
feature appears in PySide, IPython users should use
the PyQT4 wrapper for QT4, which remains the matplotlib default.

An rcParam entry, "backend.qt4", has been added to allow users
to select PyQt4, PyQt4v2, or PySide.  The latter two use the
Version 2 Qt API.  In most cases, users can ignore this rcParam
variable; it is available to aid in testing, and to provide control
for users who are embedding matplotlib in a PyQt4 or PySide app.


Legend
------

Jae-Joon Lee has improved plot legends. First,
legends for complex plots such as :meth:`~matplotlib.pyplot.stem` plots
will now display correctly. Second, the 'best' placement of a legend has
been improved in the presence of NANs.

See :ref:`legend-complex-plots` for more detailed explanation and
examples.

.. plot:: mpl_examples/pylab_examples/legend_demo4.py

mplot3d
-------

In continuing the efforts to make 3D plotting in matplotlib just as easy
as 2D plotting, Ben Root has made several improvements to the
:mod:`~mpl_toolkits.mplot3d` module.

* :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` has been
  improved to bring the class towards feature-parity with regular
  Axes objects

* Documentation for :ref:`toolkit_mplot3d-index` was significantly expanded

* Axis labels and orientation improved

* Most 3D plotting functions now support empty inputs

* Ticker offset display added:

.. plot:: mpl_examples/mplot3d/offset_demo.py

* :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.contourf`
  gains *zdir* and *offset* kwargs. You can now do this:

.. plot:: mpl_examples/mplot3d/contourf3d_demo2.py

Numerix support removed
-----------------------

After more than two years of deprecation warnings, Numerix support has
now been completely removed from matplotlib.

Markers
-------

The list of available markers for :meth:`~matplotlib.pyplot.plot` and
:meth:`~matplotlib.pyplot.scatter` has now been merged. While they
were mostly similar, some markers existed for one function, but not
the other. This merge did result in a conflict for the 'd' diamond
marker. Now, 'd' will be interpreted to always mean "thin" diamond
while 'D' will mean "regular" diamond.

Thanks to Michael Droettboom for this effort.

Other improvements
------------------

* Unit support for polar axes and :func:`~matplotlib.axes.Axes.arrow`

* :class:`~matplotlib.projections.polar.PolarAxes` gains getters and setters for
  "theta_direction", and "theta_offset" to allow for theta to go in
  either the clock-wise or counter-clockwise direction and to specify where zero
  degrees should be placed.
  :meth:`~matplotlib.projections.polar.PolarAxes.set_theta_zero_location` is an
  added convenience function.

* Fixed error in argument handling for tri-functions such as
  :meth:`~matplotlib.pyplot.tripcolor`

* ``axes.labelweight`` parameter added to rcParams.

* For :meth:`~matplotlib.pyplot.imshow`, *interpolation='nearest'* will
  now always perform an interpolation. A "none" option has been added to
  indicate no interpolation at all.

* An error in the Hammer projection has been fixed.

* *clabel* for :meth:`~matplotlib.pyplot.contour` now accepts a callable.
  Thanks to Daniel Hyams for the original patch.

* Jae-Joon Lee added the :class:`~mpl_toolkits.axes_grid1.axes_divider.HBox`
  and :class:`~mpl_toolkits.axes_grid1.axes_divider.VBox` classes.

* Christoph Gohlke reduced memory usage in :meth:`~matplotlib.pyplot.imshow`.

* :meth:`~matplotlib.pyplot.scatter` now accepts empty inputs.

* The behavior for 'symlog' scale has been fixed, but this may result
  in some minor changes to existing plots.  This work was refined by
  ssyr.

* Peter Butterworth added named figure support to
  :func:`~matplotlib.pyplot.figure`.

* Michiel de Hoon has modified the MacOSX backend to make
  its interactive behavior consistent with the other backends.

* Pim Schellart added a new colormap called "cubehelix".
  Sameer Grover also added a colormap called "coolwarm". See it and all
  other colormaps :ref:`here <pylab_examples-show_colormaps>`.

* Many bug fixes and documentation improvements.

.. _whats-new-1-0:

new in matplotlib-1.0
======================

.. _whats-new-html5:

HTML5/Canvas backend
---------------------

Simon Ratcliffe and Ludwig Schwardt have released an `HTML5/Canvas
<http://code.google.com/p/mplh5canvas/>`_ backend for matplotlib.  The
backend is almost feature complete, and they have done a lot of work
comparing their html5 rendered images with our core renderer Agg.  The
backend features client/server interactive navigation of matplotlib
figures in an html5 compliant browser.

Sophisticated subplot grid layout
---------------------------------

Jae-Joon Lee has written :mod:`~matplotlib.gridspec`, a new module for
doing complex subplot layouts, featuring row and column spans and
more.  See :ref:`gridspec-guide` for a tutorial overview.

.. plot:: users/plotting/examples/demo_gridspec01.py

Easy pythonic subplots
-----------------------

Fernando Perez got tired of all the boilerplate code needed to create a
figure and multiple subplots when using the matplotlib API, and wrote
a :func:`~matplotlib.pyplot.subplots` helper function.  Basic usage
allows you to create the figure and an array of subplots with numpy
indexing (starts with 0).  Eg::

  fig, axarr = plt.subplots(2, 2)
  axarr[0,0].plot([1,2,3])   # upper, left

See :ref:`pylab_examples-subplots_demo` for several code examples.

Contour fixes and and triplot
---------------------------------

Ian Thomas has fixed a long-standing bug that has vexed our most
talented developers for years.  :func:`~matplotlib.pyplot.contourf`
now handles interior masked regions, and the boundaries of line and
filled contours coincide.

Additionally, he has contributed a new module :mod:`~matplotlib.tri` and
helper function :func:`~matplotlib.pyplot.triplot` for creating and
plotting unstructured triangular grids.

.. plot:: mpl_examples/pylab_examples/triplot_demo.py

multiple calls to show supported
---------------------------------

A long standing request is to support multiple calls to
:func:`~matplotlib.pyplot.show`.  This has been difficult because it
is hard to get consistent behavior across operating systems, user
interface toolkits and versions.  Eric Firing has done a lot of work
on rationalizing show across backends, with the desired behavior to
make show raise all newly created figures and block execution until
they are closed.  Repeated calls to show should raise newly created
figures since the last call.  Eric has done a lot of testing on the
user interface toolkits and versions and platforms he has access to,
but it is not possible to test them all, so please report problems to
the `mailing list
<http://sourceforge.net/mailarchive/forum.php?forum_name=matplotlib-users>`_
and `bug tracker
<http://sourceforge.net/tracker/?group_id=80706&atid=560720>`_.


mplot3d graphs can be embedded in arbitrary axes
-------------------------------------------------

You can now place an mplot3d graph into an arbitrary axes location,
supporting mixing of 2D and 3D graphs in the same figure, and/or
multiple 3D graphs in a single figure, using the "projection" keyword
argument to add_axes or add_subplot.  Thanks Ben Root.

.. plot:: pyplots/whats_new_1_subplot3d.py

tick_params
------------

Eric Firing wrote tick_params, a convenience method for changing the
appearance of ticks and tick labels. See pyplot function
:func:`~matplotlib.pyplot.tick_params` and associated Axes method
:meth:`~matplotlib.axes.Axes.tick_params`.

Lots of performance and feature enhancements
---------------------------------------------


* Faster magnification of large images, and the ability to zoom in to
  a single pixel

* Local installs of documentation work better

* Improved "widgets" -- mouse grabbing is supported

* More accurate snapping of lines to pixel boundaries

* More consistent handling of color, particularly the alpha channel,
  throughout the API

Much improved software carpentry
---------------------------------

The matplotlib trunk is probably in as good a shape as it has ever
been, thanks to improved `software carpentry
<http://software-carpentry.org/>`_.  We now have a `buildbot
<http://buildbot.net/trac>`_ which runs a suite of `nose
<http://code.google.com/p/python-nose/>`_ regression tests on every
svn commit, auto-generating a set of images and comparing them against
a set of known-goods, sending emails to developers on failures with a
pixel-by-pixel `image comparison
<http://mpl.code.astraw.com/overview.html>`_.  Releases and release
bugfixes happen in branches, allowing active new feature development
to happen in the trunk while keeping the release branches stable.
Thanks to Andrew Straw, Michael Droettboom and other matplotlib
developers for the heavy lifting.

Bugfix marathon
----------------

Eric Firing went on a bug fixing and closing marathon, closing over
100 bugs on the `bug tracker
<http://sourceforge.net/tracker/?group_id=80706&atid=560720>`_ with
help from Jae-Joon Lee, Michael Droettboom, Christoph Gohlke and
Michiel de Hoon.


.. _whats-new-0-99:

new in matplotlib-0.99
======================



New documentation
-----------------

Jae-Joon Lee has written two new guides :ref:`plotting-guide-legend`
and :ref:`plotting-guide-annotation`.  Michael Sarahan has written
:ref:`image_tutorial`.  John Hunter has written two new tutorials on
working with paths and transformations: :ref:`path_tutorial` and
:ref:`transforms_tutorial`.

.. _whats-new-mplot3d:

mplot3d
--------


Reinier Heeres has ported John Porter's mplot3d over to the new
matplotlib transformations framework, and it is now available as a
toolkit mpl_toolkits.mplot3d (which now comes standard with all mpl
installs).  See :ref:`mplot3d-examples-index` and
:ref:`toolkit_mplot3d-tutorial`

.. plot:: pyplots/whats_new_99_mplot3d.py

.. _whats-new-axes-grid:

axes grid toolkit
-----------------

Jae-Joon Lee has added a new toolkit to ease displaying multiple images in
matplotlib, as well as some support for curvilinear grids to support
the world coordinate system. The toolkit is included standard with all
new mpl installs.  See :ref:`axes_grid-examples-index` and
:ref:`axes_grid_users-guide-index`.

.. plot:: pyplots/whats_new_99_axes_grid.py

.. _whats-new-spine:

Axis spine placement
--------------------

Andrew Straw has added the ability to place "axis spines" -- the lines
that denote the data limits -- in various arbitrary locations.  No
longer are your axis lines constrained to be a simple rectangle around
the figure -- you can turn on or off left, bottom, right and top, as
well as "detach" the spine to offset it away from the data.  See
:ref:`pylab_examples-spine_placement_demo` and
:class:`matplotlib.spines.Spine`.

.. plot:: pyplots/whats_new_99_spines.py


.. _whats-new-0-98-4:

new in 0.98.4
=============

It's been four months since the last matplotlib release, and there are
a lot of new features and bug-fixes.

Thanks to Charlie Moad for testing and preparing the source release,
including binaries for OS X and Windows for python 2.4 and 2.5 (2.6
and 3.0 will not be available until numpy is available on those
releases).  Thanks to the many developers who contributed to this
release, with contributions from Jae-Joon Lee, Michael Droettboom,
Ryan May, Eric Firing, Manuel Metz, Jouni K. Seppänen, Jeff Whitaker,
Darren Dale, David Kaplan, Michiel de Hoon and many others who
submitted patches

.. _legend-refactor:

Legend enhancements
--------------------

Jae-Joon has rewritten the legend class, and added support for
multiple columns and rows, as well as fancy box drawing.  See
:func:`~matplotlib.pyplot.legend` and
:class:`matplotlib.legend.Legend`.

.. plot:: pyplots/whats_new_98_4_legend.py

.. _fancy-annotations:

Fancy annotations and arrows
-----------------------------

Jae-Joon has added lot's of support to annotations for drawing fancy
boxes and connectors in annotations.  See
:func:`~matplotlib.pyplot.annotate` and
:class:`~matplotlib.patches.BoxStyle`,
:class:`~matplotlib.patches.ArrowStyle`, and
:class:`~matplotlib.patches.ConnectionStyle`.

.. plot:: pyplots/whats_new_98_4_fancy.py

.. _psd-amplitude:


Native OS X backend
--------------------

Michiel de Hoon has provided a native Mac OSX backend that is almost
completely implemented in C. The backend can therefore use Quartz
directly and, depending on the application, can be orders of magnitude
faster than the existing backends. In addition, no third-party
libraries are needed other than Python and NumPy. The backend is
interactive from the usual terminal application on Mac using regular
Python. It hasn't been tested with ipython yet, but in principle it
should to work there as well.  Set 'backend : macosx' in your
matplotlibrc file, or run your script with::

    > python myfile.py -dmacosx

psd amplitude scaling
-------------------------

Ryan May did a lot of work to rationalize the amplitude scaling of
:func:`~matplotlib.pyplot.psd` and friends.  See
:ref:`pylab_examples-psd_demo2`. and :ref:`pylab_examples-psd_demo3`.
The changes should increase MATLAB
compatabililty and increase scaling options.

.. _fill-between:

Fill between
------------------

Added a :func:`~matplotlib.pyplot.fill_between` function to make it
easier to do shaded region plots in the presence of masked data.  You
can pass an *x* array and a *ylower* and *yupper* array to fill
betweem, and an optional *where* argument which is a logical mask
where you want to do the filling.

.. plot:: pyplots/whats_new_98_4_fill_between.py

Lots more
-----------

Here are the 0.98.4 notes from the CHANGELOG::

    Added mdehoon's native macosx backend from sf patch 2179017 - JDH

    Removed the prints in the set_*style commands.  Return the list of
    pprinted strings instead - JDH

    Some of the changes Michael made to improve the output of the
    property tables in the rest docs broke of made difficult to use
    some of the interactive doc helpers, eg setp and getp.  Having all
    the rest markup in the ipython shell also confused the docstrings.
    I added a new rc param docstring.harcopy, to format the docstrings
    differently for hardcopy and other use.  Ther ArtistInspector
    could use a little refactoring now since there is duplication of
    effort between the rest out put and the non-rest output - JDH

    Updated spectral methods (psd, csd, etc.) to scale one-sided
    densities by a factor of 2 and, optionally, scale all densities by
    the sampling frequency.  This gives better MATLAB
    compatibility. -RM

    Fixed alignment of ticks in colorbars. -MGD

    drop the deprecated "new" keyword of np.histogram() for numpy 1.2
    or later.  -JJL

    Fixed a bug in svg backend that new_figure_manager() ignores
    keywords arguments such as figsize, etc. -JJL

    Fixed a bug that the handlelength of the new legend class set too
    short when numpoints=1 -JJL

    Added support for data with units (e.g. dates) to
    Axes.fill_between. -RM

    Added fancybox keyword to legend. Also applied some changes for
    better look, including baseline adjustment of the multiline texts
    so that it is center aligned. -JJL

    The transmuter classes in the patches.py are reorganized as
    subclasses of the Style classes. A few more box and arrow styles
    are added. -JJL

    Fixed a bug in the new legend class that didn't allowed a tuple of
    coordinate vlaues as loc. -JJL

    Improve checks for external dependencies, using subprocess
    (instead of deprecated popen*) and distutils (for version
    checking) - DSD

    Reimplementaion of the legend which supports baseline alignement,
    multi-column, and expand mode. - JJL

    Fixed histogram autoscaling bug when bins or range are given
    explicitly (fixes Debian bug 503148) - MM

    Added rcParam axes.unicode_minus which allows plain hypen for
    minus when False - JDH

    Added scatterpoints support in Legend. patch by Erik Tollerud -
    JJL

    Fix crash in log ticking. - MGD

    Added static helper method BrokenHBarCollection.span_where and
    Axes/pyplot method fill_between.  See
    examples/pylab/fill_between.py - JDH

    Add x_isdata and y_isdata attributes to Artist instances, and use
    them to determine whether either or both coordinates are used when
    updating dataLim.  This is used to fix autoscaling problems that
    had been triggered by axhline, axhspan, axvline, axvspan. - EF

    Update the psd(), csd(), cohere(), and specgram() methods of Axes
    and the csd() cohere(), and specgram() functions in mlab to be in
    sync with the changes to psd().  In fact, under the hood, these
    all call the same core to do computations. - RM

    Add 'pad_to' and 'sides' parameters to mlab.psd() to allow
    controlling of zero padding and returning of negative frequency
    components, respecitively.  These are added in a way that does not
    change the API. - RM

    Fix handling of c kwarg by scatter; generalize is_string_like to
    accept numpy and numpy.ma string array scalars. - RM and EF

    Fix a possible EINTR problem in dviread, which might help when
    saving pdf files from the qt backend. - JKS

    Fix bug with zoom to rectangle and twin axes - MGD

    Added Jae Joon's fancy arrow, box and annotation enhancements --
    see examples/pylab_examples/annotation_demo2.py

    Autoscaling is now supported with shared axes - EF

    Fixed exception in dviread that happened with Minion - JKS

    set_xlim, ylim now return a copy of the viewlim array to avoid
    modify inplace surprises

    Added image thumbnail generating function
    matplotlib.image.thumbnail.  See examples/misc/image_thumbnail.py
    - JDH

    Applied scatleg patch based on ideas and work by Erik Tollerud and
    Jae-Joon Lee. - MM

    Fixed bug in pdf backend: if you pass a file object for output
    instead of a filename, e.g. in a wep app, we now flush the object
    at the end. - JKS

    Add path simplification support to paths with gaps. - EF

    Fix problem with AFM files that don't specify the font's full name
    or family name. - JKS

    Added 'scilimits' kwarg to Axes.ticklabel_format() method, for
    easy access to the set_powerlimits method of the major
    ScalarFormatter. - EF

    Experimental new kwarg borderpad to replace pad in legend, based
    on suggestion by Jae-Joon Lee.  - EF

    Allow spy to ignore zero values in sparse arrays, based on patch
    by Tony Yu.  Also fixed plot to handle empty data arrays, and
    fixed handling of markers in figlegend. - EF

    Introduce drawstyles for lines. Transparently split linestyles
    like 'steps--' into drawstyle 'steps' and linestyle '--'.  Legends
    always use drawstyle 'default'. - MM

    Fixed quiver and quiverkey bugs (failure to scale properly when
    resizing) and added additional methods for determining the arrow
    angles - EF

    Fix polar interpolation to handle negative values of theta - MGD

    Reorganized cbook and mlab methods related to numerical
    calculations that have little to do with the goals of those two
    modules into a separate module numerical_methods.py Also, added
    ability to select points and stop point selection with keyboard in
    ginput and manual contour labeling code.  Finally, fixed contour
    labeling bug. - DMK

    Fix backtick in Postscript output. - MGD

    [ 2089958 ] Path simplification for vector output backends
    Leverage the simplification code exposed through path_to_polygons
    to simplify certain well-behaved paths in the vector backends
    (PDF, PS and SVG).  "path.simplify" must be set to True in
    matplotlibrc for this to work.  - MGD

    Add "filled" kwarg to Path.intersects_path and
    Path.intersects_bbox. - MGD

    Changed full arrows slightly to avoid an xpdf rendering problem
    reported by Friedrich Hagedorn. - JKS

    Fix conversion of quadratic to cubic Bezier curves in PDF and PS
    backends. Patch by Jae-Joon Lee. - JKS

    Added 5-point star marker to plot command q- EF

    Fix hatching in PS backend - MGD

    Fix log with base 2 - MGD

    Added support for bilinear interpolation in
    NonUniformImage; patch by Gregory Lielens. - EF

    Added support for multiple histograms with data of
    different length - MM

    Fix step plots with log scale - MGD

    Fix masked arrays with markers in non-Agg backends - MGD

    Fix clip_on kwarg so it actually works correctly - MGD

    Fix locale problems in SVG backend - MGD

    fix quiver so masked values are not plotted - JSW

    improve interactive pan/zoom in qt4 backend on windows - DSD

    Fix more bugs in NaN/inf handling.  In particular, path
    simplification (which does not handle NaNs or infs) will be turned
    off automatically when infs or NaNs are present.  Also masked
    arrays are now converted to arrays with NaNs for consistent
    handling of masks and NaNs - MGD and EF
