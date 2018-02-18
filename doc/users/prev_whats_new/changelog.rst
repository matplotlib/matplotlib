.. _old_changelog:

List of changes to Matplotlib prior to 2015
===========================================

This is a list of the changes made to Matplotlib from 2003 to 2015. For more
recent changes, please refer to the `what's new <../whats_new.html>`_ or
the `API changes <../../api/api_changes.html>`_.

2015-11-16 Levels passed to contour(f) and tricontour(f) must be in increasing
           order.

2015-10-21 Added TextBox widget


2015-10-21 Added get_ticks_direction()

2015-02-27 Added the rcParam 'image.composite_image' to permit users
		   to decide whether they want the vector graphics backends to combine
		   all images within a set of axes into a single composite image.
		   (If images do not get combined, users can open vector graphics files
		   in Adobe Illustrator or Inkscape and edit each image individually.)

2015-02-19 Rewrite of C++ code that calculates contours to add support for
           corner masking.  This is controlled by the 'corner_mask' keyword
           in plotting commands 'contour' and 'contourf'. - IMT

2015-01-23 Text bounding boxes are now computed with advance width rather than
           ink area.  This may result in slightly different placement of text.

2014-10-27 Allowed selection of the backend using the `MPLBACKEND` environment
           variable. Added documentation on backend selection methods.

2014-09-27 Overhauled `colors.LightSource`.  Added `LightSource.hillshade` to
           allow the independent generation of illumination maps. Added new
           types of blending for creating more visually appealing shaded relief
           plots (e.g.  `blend_mode="overlay"`, etc, in addition to the legacy
           "hsv" mode).

2014-06-10 Added Colorbar.remove()

2014-06-07 Fixed bug so radial plots can be saved as ps in py3k.

2014-06-01 Changed the fmt kwarg of errorbar to support the
           the mpl convention that "none" means "don't draw it",
           and to default to the empty string, so that plotting
           of data points is done with the plot() function
           defaults.  Deprecated use of the None object in place
           "none".

2014-05-22 Allow the linscale keyword parameter of symlog scale to be
           smaller than one.

2014-05-20 Added logic to in FontManager to invalidate font-cache if
           if font-family rcparams have changed.

2014-05-16 Fixed the positioning of multi-line text in the PGF backend.

2014-05-14 Added Axes.add_image() as the standard way to add AxesImage
           instances to Axes. This improves the consistency with
           add_artist(), add_collection(), add_container(), add_line(),
           add_patch(), and add_table().

2014-05-02 Added colorblind-friendly colormap, named 'Wistia'.

2014-04-27 Improved input clean up in Axes.{h|v}lines
           Coerce input into a 1D ndarrays (after dealing with units).

2014-04-27 removed un-needed cast to float in stem

2014-04-23 Updated references to "ipython -pylab"
           The preferred method for invoking pylab is now using the
           "%pylab" magic.
           -Chris G.

2014-04-22 Added (re-)generate a simple automatic legend to "Figure Options"
           dialog of the Qt4Agg backend.

2014-04-22 Added an example showing the difference between
           interpolation = 'none' and interpolation = 'nearest' in
           `imshow()` when saving vector graphics files.

2014-04-22 Added violin plotting functions. See `Axes.violinplot`,
           `Axes.violin`, `cbook.violin_stats` and `mlab.GaussianKDE` for
           details.

2014-04-10 Fixed the triangular marker rendering error. The "Up" triangle was
           rendered instead of "Right" triangle and vice-versa.

2014-04-08 Fixed a bug in parasite_axes.py by making a list out
           of a generator at line 263.

2014-04-02 Added `clipon=False` to patch creation of wedges and shadows
           in `pie`.

2014-02-25 In backend_qt4agg changed from using update -> repaint under
           windows.  See comment in source near `self._priv_update` for
           longer explaination.

2014-03-27 Added tests for pie ccw parameter. Removed pdf and svg images
           from tests for pie linewidth parameter.

2014-03-24 Changed the behaviour of axes to not ignore leading or trailing
           patches of height 0 (or width 0) while calculating the x and y
           axis limits. Patches having both height == 0 and width == 0 are
           ignored.

2014-03-24 Added bool kwarg (manage_xticks) to boxplot to enable/disable
           the managemnet of the xlimits and ticks when making a boxplot.
           Default in True which maintains current behavior by default.

2014-03-23 Fixed a bug in projections/polar.py by making sure that the theta
           value being calculated when given the mouse coordinates stays within
           the range of 0 and 2 * pi.

2014-03-22 Added the keyword arguments wedgeprops and textprops to pie.
           Users can control the wedge and text properties of the pie
           in more detail, if they choose.

2014-03-17 Bug was fixed in append_axes from the AxesDivider class would not
           append axes in the right location with respect to the reference
           locator axes

2014-03-13 Add parameter 'clockwise' to function pie, True by default.

2014-02-28 Added 'origin' kwarg to `spy`

2014-02-27 Implemented separate horizontal/vertical axes padding to the
           ImageGrid in the AxesGrid toolkit

2014-02-27 Allowed markevery property of matplotlib.lines.Line2D to be, an int
           numpy fancy index, slice object, or float.  The float behaviour
           turns on markers at approximately equal display-coordinate-distances
           along the line.

2014-02-25 In backend_qt4agg changed from using update -> repaint under
           windows.  See comment in source near `self._priv_update` for
           longer explaination.

2014-01-02 `triplot` now returns the artist it adds and support of line and
           marker kwargs has been improved. GBY

2013-12-30 Made streamplot grid size consistent for different types of density
           argument. A 30x30 grid is now used for both density=1 and
           density=(1, 1).

2013-12-03 Added a pure boxplot-drawing method that allow a more complete
           customization of boxplots. It takes a list of dicts contains stats.
           Also created a function (`cbook.boxplot_stats`) that generates the
           stats needed.

2013-11-28 Added qhull extension module to perform Delaunay triangulation more
           robustly than before.  It is used by tri.Triangulation (and hence
           all pyplot.tri* methods) and mlab.griddata.  Deprecated
           matplotlib.delaunay module. - IMT

2013-11-05 Add power-law normalization method. This is useful for,
           e.g., showing small populations in a "hist2d" histogram.

2013-10-27 Added get_rlabel_position and set_rlabel_position methods to
           PolarAxes to control angular position of radial tick labels.

2013-10-06 Add stride-based functions to mlab for easy creation of 2D arrays
           with less memory.

2013-10-06 Improve window and detrend functions in mlab, particulart support for
           2D arrays.

2013-10-06 Improve performance of all spectrum-related mlab functions and plots.

2013-10-06 Added support for magnitude, phase, and angle spectrums to
           axes.specgram, and support for magnitude, phase, angle, and complex
           spectrums to mlab-specgram.

2013-10-06 Added magnitude_spectrum, angle_spectrum, and phase_spectrum plots,
           as well as magnitude_spectrum, angle_spectrum, phase_spectrum,
           and complex_spectrum functions to mlab

2013-07-12 Added support for datetime axes to 2d plots. Axis values are passed
           through Axes.convert_xunits/Axes.convert_yunits before being used by
           contour/contourf, pcolormesh and pcolor.

2013-07-12 Allowed matplotlib.dates.date2num, matplotlib.dates.num2date,
           and matplotlib.dates.datestr2num to accept n-d inputs. Also
           factored in support for n-d arrays to matplotlib.dates.DateConverter
           and matplotlib.units.Registry.

2013-06-26 Refactored the axes module: the axes module is now a folder,
           containing the following submodule:
              - _subplots.py, containing all the subplots helper methods
              - _base.py, containing several private methods and a new
                _AxesBase class. This _AxesBase class contains all the methods
                that are not directly linked to plots of the "old" Axes
              - _axes.py contains the Axes class. This class now inherits from
                _AxesBase: it contains all "plotting" methods and labelling
                methods.

           This refactoring should not affect the API. Only private methods
           are not importable from the axes module anymore.

2013-05-18 Added support for arbitrary rasterization resolutions to the
           SVG backend. Previously the resolution was hard coded to 72
           dpi. Now the backend class takes a image_dpi argument for
           its constructor, adjusts the image bounding box accordingly
           and forwards a magnification factor to the image renderer.
           The code and results now resemble those of the PDF backend.
           - MW

2013-05-08 Changed behavior of hist when given stacked=True and normed=True.
           Histograms are now stacked first, then the sum is normalized.
           Previously, each histogram was normalized, then they were stacked.

2013-04-25 Changed all instances of:

           from matplotlib import MatplotlibDeprecationWarning as mplDeprecation
           to:

           from cbook import mplDeprecation

           and removed the import into the matplotlib namespace in __init__.py
           Thomas Caswell

2013-04-15 Added 'axes.xmargin' and 'axes.ymargin' to rpParams to set default
           margins on auto-scaleing. - TAC

2013-04-16 Added patheffect support for Line2D objects.  -JJL

2013-03-31 Added support for arbitrary unstructured user-specified
           triangulations to Axes3D.tricontour[f] - Damon McDougall

2013-03-19 Added support for passing `linestyle` kwarg to `step` so all `plot`
           kwargs are passed to the underlying `plot` call.  -TAC

2013-02-25 Added classes CubicTriInterpolator, UniformTriRefiner, TriAnalyzer
           to matplotlib.tri module. - GBy

2013-01-23 Add 'savefig.directory' to rcParams to remember and fill in the last
           directory saved to for figure save dialogs - Martin Spacek

2013-01-13 Add eventplot method to axes and pyplot and EventCollection class
           to collections.

2013-01-08 Added two extra titles to axes which are flush with the left and
           right edges of the plot respectively.
           Andrew Dawson

2013-01-07 Add framealpha keyword argument to legend - PO

2013-01-16 Till Stensitzki added a baseline feature to stackplot

2012-12-22 Added classes for interpolation within triangular grids
           (LinearTriInterpolator) and to find the triangles in which points
           lie (TrapezoidMapTriFinder) to matplotlib.tri module. - IMT

2012-12-05 Added MatplotlibDeprecationWarning class for signaling deprecation.
           Matplotlib developers can use this class as follows:

           from matplotlib import MatplotlibDeprecationWarning as mplDeprecation

           In light of the fact that Python builtin DeprecationWarnings are
           ignored by default as of Python 2.7, this class was put in to allow
           for the signaling of deprecation, but via UserWarnings which are
           not ignored by default. - PI

2012-11-27 Added the *mtext* parameter for supplying matplotlib.text.Text
           instances to RendererBase.draw_tex and RendererBase.draw_text.
           This allows backends to utilize additional text attributes, like
           the alignment of text elements. - pwuertz

2012-11-26 deprecate matplotlib/mpl.py, which was used only in pylab.py and is
           now replaced by the more suitable `import matplotlib as mpl`. - PI

2012-11-25 Make rc_context available via pyplot interface - PI

2012-11-16 plt.set_cmap no longer throws errors if there is not already
           an active colorable artist, such as an image, and just sets
           up the colormap to use from that point forward. - PI

2012-11-16 Added the funcction _get_rbga_face, which is identical to
           _get_rbg_face except it return a (r,g,b,a) tuble, to line2D.
           Modified Line2D.draw to use _get_rbga_face to get the markerface
           color so that any alpha set by  markerfacecolor will respected.
           - Thomas Caswell

2012-11-13 Add a symmetric log normalization class to colors.py.
           Also added some tests for the normalization class.
           Till Stensitzki

2012-11-12 Make axes.stem take at least one argument.
           Uses a default range(n) when the first arg not provided.
           Damon McDougall

2012-11-09 Make plt.subplot() without arguments act as subplot(111) - PI

2012-11-08 Replaced plt.figure and plt.subplot calls by the newer, more
           convenient single call to plt.subplots() in the documentation
           examples - PI

2012-10-05 Add support for saving animations as animated GIFs. - JVDP

2012-08-11 Fix path-closing bug in patches.Polygon, so that regardless
           of whether the path is the initial one or was subsequently
           set by set_xy(), get_xy() will return a closed path if and
           only if get_closed() is True.  Thanks to Jacob Vanderplas. - EF

2012-08-05 When a norm is passed to contourf, either or both of the
           vmin, vmax attributes of that norm are now respected.
           Formerly they were respected only if both were
           specified. In addition, vmin and/or vmax can now
           be passed to contourf directly as kwargs. - EF

2012-07-24 Contourf handles the extend kwarg by mapping the extended
           ranges outside the normed 0-1 range so that they are
           handled by colormap colors determined by the set_under
           and set_over methods.  Previously the extended ranges
           were mapped to 0 or 1 so that the "under" and "over"
           colormap colors were ignored. This change also increases
           slightly the color contrast for a given set of contour
           levels. - EF

2012-06-24 Make use of mathtext in tick labels configurable - DSD

2012-06-05 Images loaded through PIL are now ordered correctly - CG

2012-06-02 Add new Axes method and pyplot function, hist2d. - PO

2012-05-31 Remove support for 'cairo.<format>' style of backend specification.
           Deprecate 'cairo.format' and 'savefig.extension' rcParams and
           replace with 'savefig.format'. - Martin Spacek

2012-05-29 pcolormesh now obeys the passed in "edgecolor" kwarg.
           To support this, the "shading" argument to pcolormesh now only
           takes "flat" or "gouraud".  To achieve the old "faceted" behavior,
           pass "edgecolors='k'". - MGD

2012-05-22 Added radius kwarg to pie charts. - HH

2012-05-22 Collections now have a setting "offset_position" to select whether
           the offsets are given in "screen" coordinates (default,
           following the old behavior) or "data" coordinates.  This is currently
           used internally to improve the performance of hexbin.

           As a result, the "draw_path_collection" backend methods have grown
           a new argument "offset_position". - MGD

2012-05-04 Add a new argument to pie charts - startingangle - that
           allows one to specify the angle offset for the first wedge
           of the chart. - EP

2012-05-03 symlog scale now obeys the logarithmic base.  Previously, it was
           completely ignored and always treated as base e. - MGD

2012-05-03 Allow linscalex/y keyword to symlog scale that allows the size of
           the linear portion relative to the logarithmic portion to be
           adjusted. - MGD

2012-04-14 Added new plot style: stackplot. This new feature supports stacked
           area plots. - Damon McDougall

2012-04-06 When path clipping changes a LINETO to a MOVETO, it also
           changes any CLOSEPOLY command to a LINETO to the initial
           point. This fixes a problem with pdf and svg where the
           CLOSEPOLY would then draw a line to the latest MOVETO
           position instead of the intended initial position. - JKS

2012-03-27 Add support to ImageGrid for placing colorbars only at
           one edge of each column/row. - RMM

2012-03-07 Refactor movie writing into useful classes that make use
           of pipes to write image data to ffmpeg or mencoder. Also
           improve settings for these and the ability to pass custom
           options. - RMM

2012-02-29 errorevery keyword added to errorbar to enable errorbar
           subsampling. fixes issue #600.

2012-02-28 Added plot_trisurf to the mplot3d toolkit. This supports plotting
           three dimensional surfaces on an irregular grid. - Damon McDougall

2012-01-23 The radius labels in polar plots no longer use a fixed
           padding, but use a different alignment depending on the
           quadrant they are in.  This fixes numerical problems when
           (rmax - rmin) gets too small. - MGD

2012-01-08 Add axes.streamplot to plot streamlines of a velocity field.
                   Adapted from Tom Flannaghan streamplot implementation. -TSY

2011-12-29 ps and pdf markers are now stroked only if the line width
           is nonzero for consistency with agg, fixes issue #621. - JKS

2011-12-27 Work around an EINTR bug in some versions of subprocess. - JKS

2011-10-25 added support for \operatorname to mathtext,
           including the ability to insert spaces, such as
           $\operatorname{arg\,max}$ - PI

2011-08-18 Change api of Axes.get_tightbbox and add an optional
           keyword parameter *call_axes_locator*. - JJL

2011-07-29 A new rcParam "axes.formatter.use_locale" was added, that,
           when True, will use the current locale to format tick
           labels.  This means that, for example, in the fr_FR locale,
           ',' will be used as a decimal separator.  - MGD

2011-07-15 The set of markers available in the plot() and scatter()
           commands has been unified.  In general, this gives more
           options to both than were previously available, however,
           there is one backward-incompatible change to the markers in
           scatter:

              "d" used to mean "diamond", it now means "narrow
              diamond".  "D" can be used for a "diamond".

           -MGD

2011-07-13 Fix numerical problems in symlog scale, particularly when
           linthresh <= 1.0.  Symlog plots may look different if one
           was depending on the old broken behavior - MGD

2011-07-10 Fixed argument handling error in tripcolor/triplot/tricontour,
           issue #203. - IMT

2011-07-08 Many functions added to mplot3d.axes3d to bring Axes3D
           objects more feature-parity with regular Axes objects.
           Significant revisions to the documentation as well.
           - BVR

2011-07-07 Added compatibility with IPython strategy for picking
           a version of Qt4 support, and an rcParam for making
           the choice explicitly: backend.qt4. - EF

2011-07-07 Modified AutoMinorLocator to improve automatic choice of
           the number of minor intervals per major interval, and
           to allow one to specify this number via a kwarg. - EF

2011-06-28 3D versions of scatter, plot, plot_wireframe, plot_surface,
           bar3d, and some other functions now support empty inputs. - BVR

2011-06-22 Add set_theta_offset, set_theta_direction and
           set_theta_zero_location to polar axes to control the
           location of 0 and directionality of theta. - MGD

2011-06-22 Add axes.labelweight parameter to set font weight to axis
           labels - MGD.

2011-06-20 Add pause function to pyplot. - EF

2011-06-16 Added *bottom* keyword parameter for the stem command.
           Also, implemented a legend handler for the stem plot.
           - JJL

2011-06-16 Added legend.frameon rcParams. - Mike Kaufman

2011-05-31 Made backend_qt4 compatible with PySide . - Gerald Storer

2011-04-17 Disable keyboard auto-repeat in qt4 backend by ignoring
           key events resulting from auto-repeat.  This makes
           constrained zoom/pan work. - EF

2011-04-14 interpolation="nearest" always interpolate images. A new
           mode "none" is introduced for no interpolation - JJL

2011-04-03 Fixed broken pick interface to AsteriskCollection objects
           used by scatter. - EF

2011-04-01 The plot directive Sphinx extension now supports all of the
           features in the Numpy fork of that extension.  These
           include doctest formatting, an 'include-source' option, and
           a number of new configuration options. - MGD

2011-03-29 Wrapped ViewVCCachedServer definition in a factory function.
           This class now inherits from urllib2.HTTPSHandler in order
           to fetch data from github, but HTTPSHandler is not defined
           if python was built without SSL support. - DSD

2011-03-10 Update pytz version to 2011c, thanks to Simon Cross. - JKS

2011-03-06 Add standalone tests.py test runner script. - JKS

2011-03-06 Set edgecolor to 'face' for scatter asterisk-type
           symbols; this fixes a bug in which these symbols were
           not responding to the c kwarg.  The symbols have no
           face area, so only the edgecolor is visible. - EF

2011-02-27 Support libpng version 1.5.x; suggestion by Michael
           Albert. Changed installation specification to a
           minimum of libpng version 1.2.  - EF

2011-02-20 clabel accepts a callable as an fmt kwarg; modified
           patch by Daniel Hyams. - EF

2011-02-18 scatter([], []) is now valid.  Also fixed issues
           with empty collections - BVR

2011-02-07 Quick workaround for dviread bug #3175113 - JKS

2011-02-05 Add cbook memory monitoring for Windows, using
           tasklist. - EF

2011-02-05 Speed up Normalize and LogNorm by using in-place
           operations and by using float32 for float32 inputs
           and for ints of 2 bytes or shorter; based on
           patch by Christoph Gohlke. - EF

2011-02-04 Changed imshow to use rgba as uint8 from start to
           finish, instead of going through an intermediate
           step as double precision; thanks to Christoph Gohlke. - EF

2011-01-13 Added zdir and offset arguments to contourf3d to
           bring contourf3d in feature parity with contour3d. - BVR

2011-01-04 Tag 1.0.1 for release at r8896

2011-01-03 Added display of ticker offset to 3d plots. - BVR

2011-01-03 Turn off tick labeling on interior subplots for
           pyplots.subplots when sharex/sharey is True. - JDH

2010-12-29 Implement axes_divider.HBox and VBox. -JJL


2010-11-22 Fixed error with Hammer projection. - BVR

2010-11-12 Fixed the placement and angle of axis labels in 3D plots. - BVR

2010-11-07 New rc parameters examples.download and examples.directory
           allow bypassing the download mechanism in get_sample_data.
           - JKS

2010-10-04 Fix JPEG saving bug: only accept the kwargs documented
           by PIL for JPEG files. - JKS

2010-09-15 Remove unused _wxagg extension and numerix.h. - EF

2010-08-25 Add new framework for doing animations with examples.- RM

2010-08-21 Remove unused and inappropriate methods from Tick classes:
           set_view_interval, get_minpos, and get_data_interval are
           properly found in the Axis class and don't need to be
           duplicated in XTick and YTick. - EF

2010-08-21 Change Axis.set_view_interval() so that when updating an
           existing interval, it respects the orientation of that
           interval, and can enlarge but not reduce the interval.
           This fixes a bug in which Axis.set_ticks would
           change the view limits of an inverted axis. Whether
           set_ticks should be affecting the viewLim at all remains
           an open question. - EF

2010-08-16 Handle NaN's correctly in path analysis routines.  Fixes a
           bug where the best location for a legend was not calculated
           correctly when the line contains NaNs. - MGD

2010-08-14 Fix bug in patch alpha handling, and in bar color kwarg - EF

2010-08-12 Removed all traces of numerix module after 17 months of
           deprecation warnings. - EF

2010-08-05 Added keyword arguments 'thetaunits' and 'runits' for polar
           plots.  Fixed PolarAxes so that when it set default
           Formatters, it marked them as such.  Fixed semilogx and
           semilogy to no longer blindly reset the ticker information
           on the non-log axis.  Axes.arrow can now accept unitized
           data. - JRE

2010-08-03 Add support for MPLSETUPCFG variable for custom setup.cfg
           filename.  Used by sage buildbot to build an mpl w/ no gui
           support - JDH

2010-08-01 Create directory specified by MPLCONFIGDIR if it does
           not exist. - ADS

2010-07-20 Return Qt4's default cursor when leaving the canvas - DSD

2010-07-06 Tagging for mpl 1.0 at r8502


2010-07-05 Added Ben Root's patch to put 3D plots in arbitrary axes,
           allowing you to mix 3d and 2d in different axes/subplots or
           to have multiple 3D plots in one figure.  See
           examples/mplot3d/subplot3d_demo.py - JDH

2010-07-05 Preferred kwarg names in set_xlim are now 'left' and
           'right'; in set_ylim, 'bottom' and 'top'; original
           kwargs are still accepted without complaint. - EF

2010-07-05 TkAgg and FltkAgg backends are now consistent with other
           interactive backends: when used in scripts from the
           command line (not from ipython -pylab), show blocks,
           and can be called more than once. - EF

2010-07-02 Modified CXX/WrapPython.h to fix "swab bug" on solaris so
           mpl can compile on Solaris with CXX6 in the trunk.  Closes
           tracker bug 3022815 - JDH

2010-06-30 Added autoscale convenience method and corresponding
           pyplot function for simplified control of autoscaling;
           and changed axis, set_xlim, and set_ylim so that by
           default, they turn off the autoscaling on the relevant
           axis or axes.  Therefore one can call set_xlim before
           plotting a line, for example, and the limits will be
           retained. - EF

2010-06-20 Added Axes.tick_params and corresponding pyplot function
           to control tick and tick label appearance after an Axes
           has been created. - EF

2010-06-09 Allow Axes.grid to control minor gridlines; allow
           Axes.grid and Axis.grid to control major and minor
           gridlines in the same method call. - EF

2010-06-06 Change the way we do split/dividend adjustments in
           finance.py to handle dividends and fix the zero division bug reported
           in sf bug 2949906 and 2123566.  Note that volume is not adjusted
           because the Yahoo CSV does not distinguish between share
           split and dividend adjustments making it near impossible to
           get volume adjustement right (unless we want to guess based
           on the size of the adjustment or scrape the html tables,
           which we don't) - JDH

2010-06-06 Updated dateutil to 1.5 and pytz to 2010h.

2010-06-02 Add error_kw kwarg to Axes.bar(). - EF

2010-06-01 Fix pcolormesh() and QuadMesh to pass on kwargs as
           appropriate. - RM

2010-05-18 Merge mpl_toolkits.gridspec into the main tree. - JJL

2010-05-04 Improve backend_qt4 so it displays figures with the
           correct size - DSD

2010-04-20 Added generic support for connecting to a timer for events. This
           adds TimerBase, TimerGTK, TimerQT, TimerWx, and TimerTk to
           the backends and a new_timer() method to each backend's
           canvas to allow ease of creating a new timer. - RM

2010-04-20 Added margins() Axes method and pyplot function. - EF

2010-04-18 update the axes_grid documentation. -JJL

2010-04-18 Control MaxNLocator parameters after instantiation,
           and via Axes.locator_params method, with corresponding
           pyplot function. -EF

2010-04-18 Control ScalarFormatter offsets directly and via the
           Axes.ticklabel_format() method, and add that to pyplot. -EF

2010-04-16 Add a close_event to the backends. -RM

2010-04-06 modify axes_grid examples to use axes_grid1 and axisartist. -JJL

2010-04-06 rebase axes_grid using axes_grid1 and axisartist modules. -JJL

2010-04-06 axes_grid toolkit is splitted into two separate modules,
           axes_grid1 and axisartist. -JJL

2010-04-05 Speed up import: import pytz only if and when it is
           needed.  It is not needed if the rc timezone is UTC. - EF

2010-04-03 Added color kwarg to Axes.hist(), based on work by
           Jeff Klukas. - EF

2010-03-24 refactor colorbar code so that no cla() is necessary when
           mappable is changed. -JJL

2010-03-22 fix incorrect rubber band during the zoom mode when mouse
           leaves the axes. -JJL

2010-03-21 x/y key during the zoom mode only changes the x/y limits. -JJL

2010-03-20 Added pyplot.sca() function suggested by JJL. - EF

2010-03-20 Added conditional support for new Tooltip API in gtk backend. - EF

2010-03-20 Changed plt.fig_subplot() to plt.subplots() after discussion on
           list, and changed its API to return axes as a numpy object array
           (with control of dimensions via squeeze keyword). FP.

2010-03-13 Manually brought in commits from branch::

    ------------------------------------------------------------------------
    r8191 | leejjoon | 2010-03-13 17:27:57 -0500 (Sat, 13 Mar 2010) | 1 line

  fix the bug that handles for scatter are incorrectly set when dpi!=72.
  Thanks to Ray Speth for the bug report.


2010-03-03 Manually brought in commits from branch via diff/patch (svnmerge is broken)::

    ------------------------------------------------------------------------
    r8175 | leejjoon | 2010-03-03 10:03:30 -0800 (Wed, 03 Mar 2010) | 1 line

    fix arguments of allow_rasterization.draw_wrapper
    ------------------------------------------------------------------------
    r8174 | jdh2358 | 2010-03-03 09:15:58 -0800 (Wed, 03 Mar 2010) | 1 line

    added support for favicon in docs build
    ------------------------------------------------------------------------
    r8173 | jdh2358 | 2010-03-03 08:56:16 -0800 (Wed, 03 Mar 2010) | 1 line

    applied Mattias get_bounds patch
    ------------------------------------------------------------------------
    r8172 | jdh2358 | 2010-03-03 08:31:42 -0800 (Wed, 03 Mar 2010) | 1 line

    fix svnmerge download instructions
    ------------------------------------------------------------------------
    r8171 | jdh2358 | 2010-03-03 07:47:48 -0800 (Wed, 03 Mar 2010) | 1 line



2010-02-25 add annotation_demo3.py that demonstrates new functionality. -JJL

2010-02-25 refactor Annotation to support arbitrary Transform as xycoords
           or textcoords. Also, if a tuple of two coordinates is provided,
           they are interpreted as coordinates for each x and y position.
           -JJL

2010-02-24 Added pyplot.fig_subplot(), to create a figure and a group of
           subplots in a single call.  This offers an easier pattern than
           manually making figures and calling add_subplot() multiple times. FP

2010-02-17 Added Gokhan's and Mattias' customizable keybindings patch
           for the toolbar.  You can now set the keymap.* properties
           in the matplotlibrc file.  Newbindings were added for
           toggling log scaling on the x-axis. JDH

2010-02-16 Committed TJ's filled marker patch for
           left|right|bottom|top|full filled markers.  See
           examples/pylab_examples/filledmarker_demo.py. JDH

2010-02-11 Added 'bootstrap' option to boxplot. This allows bootstrap
           estimates of median confidence intervals. Based on an
           initial patch by Paul Hobson. - ADS

2010-02-06 Added setup.cfg "basedirlist" option to override setting
           in setupext.py "basedir" dictionary; added "gnu0"
           platform requested by Benjamin Drung. - EF

2010-02-06 Added 'xy' scaling option to EllipseCollection. - EF

2010-02-03 Made plot_directive use a custom PlotWarning category, so that
           warnings can be turned into fatal errors easily if desired. - FP

2010-01-29 Added draggable method to Legend to allow mouse drag
           placement.  Thanks Adam Fraser. JDH

2010-01-25 Fixed a bug reported by Olle Engdegard, when using
           histograms with stepfilled and log=True - MM

2010-01-16 Upgraded CXX to 6.1.1 - JDH

2009-01-16 Don't create minor ticks on top of existing major
           ticks. Patch by Neil Crighton. -ADS

2009-01-16 Ensure three minor ticks always drawn (SF# 2924245). Patch
           by Neil Crighton. -ADS

2010-01-16 Applied patch by Ian Thomas to fix two contouring
           problems: now contourf handles interior masked regions,
           and the boundaries of line and filled contours coincide. - EF

2009-01-11 The color of legend patch follows the rc parameters
           axes.facecolor and axes.edgecolor. -JJL

2009-01-11 adjustable of Axes can be "box-forced" which allow
           sharing axes. -JJL

2009-01-11 Add add_click and pop_click methods in
           BlockingContourLabeler. -JJL


2010-01-03 Added rcParams['axes.color_cycle'] - EF

2010-01-03 Added Pierre's qt4 formlayout editor and toolbar button - JDH

2009-12-31 Add support for using math text as marker symbols (Thanks to tcb)
           - MGD

2009-12-31 Commit a workaround for a regression in PyQt4-4.6.{0,1} - DSD

2009-12-22 Fix cmap data for gist_earth_r, etc. -JJL

2009-12-20 spines: put spines in data coordinates, add set_bounds()
           call. -ADS

2009-12-18 Don't limit notch size in boxplot to q1-q3 range, as this
           is effectively making the data look better than it is. - ADS

2009-12-18 mlab.prctile handles even-length data, such that the median
           is the mean of the two middle values. - ADS

2009-12-15 Add raw-image (unsampled) support for the ps backend. - JJL

2009-12-14 Add patch_artist kwarg to boxplot, but keep old default.
           Convert boxplot_demo2.py to use the new patch_artist. - ADS

2009-12-06 axes_grid: reimplemented AxisArtist with FloatingAxes support.
           Added new examples. - JJL

2009-12-01 Applied Laurent Dufrechou's patch to improve blitting with
           the qt4 backend - DSD

2009-11-13 The pdf backend now allows changing the contents of
           a pdf file's information dictionary via PdfPages.infodict. - JKS

2009-11-12 font_manager.py should no longer cause EINTR on Python 2.6
           (but will on the 2.5 version of subprocess). Also the
           fc-list command in that file was fixed so now it should
           actually find the list of fontconfig fonts. - JKS

2009-11-10 Single images, and all images in renderers with
           option_image_nocomposite (i.e. agg, macosx and the svg
           backend when rcParams['svg.image_noscale'] is True), are
           now drawn respecting the zorder relative to other
           artists. (Note that there may now be inconsistencies across
           backends when more than one image is drawn at varying
           zorders, but this change introduces correct behavior for
           the backends in which it's easy to do so.)

2009-10-21 Make AutoDateLocator more configurable by adding options
           to control the maximum and minimum number of ticks. Also
           add control of the intervals to be used for ticking. This
           does not change behavior but opens previously hard-coded
           behavior to runtime modification`. - RMM

2009-10-19 Add "path_effects" support for Text and Patch. See
           examples/pylab_examples/patheffect_demo.py -JJL

2009-10-19 Add "use_clabeltext" option to clabel. If True, clabels
           will be created with ClabelText class, which recalculates
           rotation angle of the label during the drawing time. -JJL

2009-10-16 Make AutoDateFormatter actually use any specified
           timezone setting.This was only working correctly
           when no timezone was specified. - RMM

2009-09-27 Beginnings of a capability to test the pdf backend. - JKS

2009-09-27 Add a savefig.extension rcparam to control the default
           filename extension used by savefig. - JKS

===============================================

2009-09-21 Tagged for release 0.99.1

2009-09-20 Fix usetex spacing errors in pdf backend. - JKS

2009-09-20 Add Sphinx extension to highlight IPython console sessions,
           originally authored (I think) by Michael Droetboom. - FP

2009-09-20 Fix off-by-one error in dviread.Tfm, and additionally protect
           against exceptions in case a dvi font is missing some metrics. - JKS

2009-09-15 Implement draw_text and draw_tex method of backend_base using
           the textpath module. Implement draw_tex method of the svg
           backend. - JJL

2009-09-15 Don't fail on AFM files containing floating-point bounding boxes - JKS

2009-09-13 AxesGrid : add modified version of colorbar. Add colorbar
           location howto. - JJL

2009-09-07 AxesGrid : implemented axisline style.
           Added a demo examples/axes_grid/demo_axisline_style.py- JJL

2009-09-04 Make the textpath class as a separate moduel
           (textpath.py). Add support for mathtext and tex.- JJL

2009-09-01 Added support for Gouraud interpolated triangles.
           pcolormesh now accepts shading='gouraud' as an option. - MGD

2009-08-29 Added matplotlib.testing package, which contains a Nose
           plugin and a decorator that lets tests be marked as
           KnownFailures - ADS

2009-08-20 Added scaled dict to AutoDateFormatter for customized
           scales - JDH

2009-08-15 Pyplot interface: the current image is now tracked at the
           figure and axes level, addressing tracker item 1656374. - EF

2009-08-15 Docstrings are now manipulated with decorators defined
           in a new module, docstring.py, thanks to Jason Coombs. - EF

2009-08-14 Add support for image filtering for agg back end. See the example
           demo_agg_filter.py. -JJL

2009-08-09 AnnotationBbox added. Similar to Annotation, but works with
           OffsetBox instead of Text. See the example
           demo_annotation_box.py. -JJL

2009-08-07 BboxImage implemented. Two examples, demo_bboximage.py and
           demo_ribbon_box.py added. - JJL

2009-08-07 In an effort to simplify the backend API, all clipping rectangles
           and paths are now passed in using GraphicsContext objects, even
           on collections and images.  Therefore:

             draw_path_collection(self, master_transform, cliprect, clippath,
                                  clippath_trans, paths, all_transforms, offsets,
                                  offsetTrans, facecolors, edgecolors, linewidths,
                                  linestyles, antialiaseds, urls)

                                             becomes:

             draw_path_collection(self, gc, master_transform, paths, all_transforms,
                                  offsets, offsetTrans, facecolors, edgecolors,
                                  linewidths, linestyles, antialiaseds, urls)



             draw_quad_mesh(self, master_transform, cliprect, clippath,
                            clippath_trans, meshWidth, meshHeight, coordinates,
                            offsets, offsetTrans, facecolors, antialiased,
                            showedges)

                                             becomes:

             draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                            coordinates, offsets, offsetTrans, facecolors,
                            antialiased, showedges)



             draw_image(self, x, y, im, bbox, clippath=None, clippath_trans=None)

                                             becomes:

             draw_image(self, gc, x, y, im)

           - MGD

2009-08-06 Tagging the 0.99.0 release at svn r7397 - JDH

           * fixed an alpha colormapping bug posted on sf 2832575

           * fix typo in axes_divider.py. use nanmin, nanmax in angle_helper.py
             (patch by Christoph Gohlke)

           * remove dup gui event in enter/leave events in gtk

           * lots of fixes for os x binaries (Thanks Russell Owen)

           * attach gtk events to mpl events -- fixes sf bug 2816580

           * applied sf patch 2815064 (middle button events for wx) and
             patch  2818092 (resize events for wx)

           * fixed boilerplate.py so it doesn't break the ReST docs.

           * removed a couple of cases of mlab.load

           * fixed rec2csv win32 file handle bug from sf patch 2831018

           * added two examples from Josh Hemann: examples/pylab_examples/barchart_demo2.py
             and examples/pylab_examples/boxplot_demo2.py

           * handled sf bugs 2831556 and 2830525; better bar error messages and
             backend driver configs

           * added miktex win32 patch from sf patch 2820194

           * apply sf patches 2830233 and 2823885 for osx setup and 64 bit;  thanks Michiel

2009-08-04 Made cbook.get_sample_data make use of the ETag and Last-Modified
           headers of mod_dav_svn. - JKS

2009-08-03 Add PathCollection; modify contourf to use complex
           paths instead of simple paths with cuts. - EF


2009-08-03 Fixed boilerplate.py so it doesn't break the ReST docs. - JKS

2009-08-03 pylab no longer provides a load and save function.  These
           are available in matplotlib.mlab, or you can use
           numpy.loadtxt and numpy.savetxt for text files, or np.save
           and np.load for binary numpy arrays. - JDH

2009-07-31 Added cbook.get_sample_data for urllib enabled fetching and
           cacheing of data needed for examples.  See
           examples/misc/sample_data_demo.py - JDH

2009-07-31 Tagging 0.99.0.rc1 at 7314 - MGD

2009-07-30 Add set_cmap and register_cmap, and improve get_cmap,
           to provide convenient handling of user-generated
           colormaps. Reorganized _cm and cm modules. - EF

2009-07-28 Quiver speed improved, thanks to tip by Ray Speth. -EF

2009-07-27 Simplify argument handling code for plot method. -EF

2009-07-25 Allow "plot(1, 2, 'r*')" to work. - EF

2009-07-22 Added an 'interp' keyword to griddata so the faster linear
           interpolation method can be chosen.  Default is 'nn', so
           default behavior (using natural neighbor method) is unchanged (JSW)

2009-07-22 Improved boilerplate.py so that it generates the correct
           signatures for pyplot functions. - JKS

2009-07-19 Fixed the docstring of Axes.step to reflect the correct
           meaning of the kwargs "pre" and "post" - See SF bug
           https://sourceforge.net/tracker/index.php?func=detail&aid=2823304&group_id=80706&atid=560720
           - JDH

2009-07-18 Fix support for hatches without color fills to pdf and svg
           backends. Add an example of that to hatch_demo.py. - JKS

2009-07-17 Removed fossils from swig version of agg backend. - EF

2009-07-14 initial submission of the annotation guide. -JJL

2009-07-14 axes_grid : minor improvements in anchored_artists and
           inset_locator. -JJL

2009-07-14 Fix a few bugs in ConnectionStyle algorithms. Add
           ConnectionPatch class. -JJL

2009-07-11 Added a fillstyle Line2D property for half filled markers
           -- see examples/pylab_examples/fillstyle_demo.py JDH

2009-07-08 Attempt to improve performance of qt4 backend, do not call
           qApp.processEvents while processing an event. Thanks Ole
           Streicher for tracking this down - DSD

2009-06-24 Add withheader option to mlab.rec2csv and changed
  use_mrecords default to False in mlab.csv2rec since this is
  partially broken - JDH

2009-06-24 backend_agg.draw_marker quantizes the main path (as in the
           draw_path). - JJL

2009-06-24 axes_grid: floating axis support added. - JJL

2009-06-14 Add new command line options to backend_driver.py to support
           running only some directories of tests - JKS

2009-06-13 partial cleanup of mlab and its importation in pylab - EF

2009-06-13 Introduce a rotation_mode property for the Text artist. See
           examples/pylab_examples/demo_text_rotation_mode.py -JJL

2009-06-07 add support for bz2 files per sf support request 2794556 -
           JDH

2009-06-06 added a properties method to the artist and inspector to
           return a dict mapping property name -> value; see sf
           feature request 2792183 - JDH

2009-06-06 added Neil's auto minor tick patch; sf patch #2789713 - JDH

2009-06-06 do not apply alpha to rgba color conversion if input is
           already rgba - JDH

2009-06-03 axes_grid : Initial check-in of curvelinear grid support. See
           examples/axes_grid/demo_curvelinear_grid.py - JJL

2009-06-01 Add set_color method to Patch - EF

2009-06-01 Spine is now derived from Patch - ADS

2009-06-01 use cbook.is_string_like() instead of isinstance() for spines - ADS

2009-06-01 cla() support for spines - ADS

2009-06-01 Removed support for gtk < 2.4. - EF

2009-05-29 Improved the animation_blit_qt4 example, which was a mix
           of the object-oriented and pylab interfaces. It is now
           strictly object-oriented - DSD

2009-05-28 Fix axes_grid toolkit to work with spine patch by ADS. - JJL

2009-05-28 Applied fbianco's patch to handle scroll wheel events in
           the qt4 backend - DSD

2009-05-26 Add support for "axis spines" to have arbitrary location. -ADS

2009-05-20 Add an empty matplotlibrc to the tests/ directory so that running
           tests will use the default set of rcparams rather than the user's
           config. - RMM

2009-05-19 Axis.grid(): allow use of which='major,minor' to have grid
           on major and minor ticks. -ADS

2009-05-18 Make psd(), csd(), and cohere() wrap properly for complex/two-sided
           versions, like specgram() (SF #2791686) - RMM

2009-05-18 Fix the linespacing bug of multiline text (#1239682). See
           examples/pylab_examples/multiline.py -JJL

2009-05-18 Add *annotation_clip* attr. for text.Annotation class.
           If True, annotation is only drawn when the annotated point is
           inside the axes area. -JJL

2009-05-17 Fix bug(#2749174) that some properties of minor ticks are
           not conserved -JJL

2009-05-17 applied Michiel's sf patch 2790638 to turn off gtk event
           loop in setupext for pygtk>=2.15.10 - JDH

2009-05-17 applied Michiel's sf patch 2792742 to speed up Cairo and
           macosx collections; speedups can be 20x.  Also fixes some
           bugs in which gc got into inconsistent state

-----------------------

2008-05-17 Release 0.98.5.3 at r7107 from the branch - JDH

2009-05-13 An optional offset and bbox support in restore_bbox.
           Add animation_blit_gtk2.py. -JJL

2009-05-13 psfrag in backend_ps now uses baseline-alignment
           when preview.sty is used ((default is
           bottom-alignment). Also, a small api imporvement
           in OffsetBox-JJL

2009-05-13 When the x-coordinate of a line is monotonically
           increasing, it is now automatically clipped at
           the stage of generating the transformed path in
           the draw method; this greatly speeds up zooming and
           panning when one is looking at a short segment of
           a long time series, for example. - EF

2009-05-11 aspect=1 in log-log plot gives square decades. -JJL

2009-05-08 clabel takes new kwarg, rightside_up; if False, labels
           will not be flipped to keep them rightside-up.  This
           allows the use of clabel to make streamfunction arrows,
           as requested by Evan Mason. - EF

2009-05-07 'labelpad' can now be passed when setting x/y labels. This
           allows controlling the spacing between the label and its
           axis. - RMM

2009-05-06 print_ps now uses mixed-mode renderer. Axes.draw rasterize
           artists whose zorder smaller than rasterization_zorder.
           -JJL

2009-05-06 Per-artist Rasterization, originally by Eric Bruning. -JJ

2009-05-05 Add an example that shows how to make a plot that updates
           using data from another process.  Thanks to Robert
           Cimrman - RMM

2009-05-05 Add Axes.get_legend_handles_labels method. - JJL

2009-05-04 Fix bug that Text.Annotation is still drawn while set to
           not visible. - JJL

2009-05-04 Added TJ's fill_betweenx patch - JDH

2009-05-02 Added options to plotfile based on question from
           Joseph Smidt and patch by Matthias Michler. - EF


2009-05-01 Changed add_artist and similar Axes methods to
           return their argument. - EF

2009-04-30 Incorrect eps bbox for landscape mode fixed - JJL

2009-04-28 Fixed incorrect bbox of eps output when usetex=True. - JJL

2009-04-24 Changed use of os.open* to instead use subprocess.Popen.
           os.popen* are deprecated in 2.6 and are removed in 3.0. - RMM

2009-04-20 Worked on axes_grid documentation. Added
           axes_grid.inset_locator. - JJL

2009-04-17 Initial check-in of the axes_grid toolkit. - JJL

2009-04-17 Added a support for bbox_to_anchor in
           offsetbox.AnchoredOffsetbox. Improved a documentation.
           - JJL

2009-04-16 Fixed a offsetbox bug that multiline texts are not
           correctly aligned.  - JJL

2009-04-16 Fixed a bug in mixed mode renderer that images produced by
           an rasterizing backend are placed with incorrect size.
           - JJL

2009-04-14 Added Jonathan Taylor's Reinier Heeres' port of John
           Porters' mplot3d to svn trunk.  Package in
           mpl_toolkits.mplot3d and demo is examples/mplot3d/demo.py.
           Thanks Reiner

2009-04-06 The pdf backend now escapes newlines and linefeeds in strings.
           Fixes sf bug #2708559; thanks to Tiago Pereira for the report.

2009-04-06 texmanager.make_dvi now raises an error if LaTeX failed to
           create an output file. Thanks to Joao Luis Silva for reporting
           this. - JKS

2009-04-05 _png.read_png() reads 12 bit PNGs (patch from
           Tobias Wood) - ADS

2009-04-04 Allow log axis scale to clip non-positive values to
           small positive value; this is useful for errorbars. - EF

2009-03-28 Make images handle nan in their array argument.
           A helper, cbook.safe_masked_invalid() was added. - EF

2009-03-25 Make contour and contourf handle nan in their Z argument. - EF

2009-03-20 Add AuxTransformBox in offsetbox.py to support some transformation.
           anchored_text.py example is enhanced and renamed
           (anchored_artists.py). - JJL

2009-03-20 Add "bar" connection style for annotation - JJL

2009-03-17 Fix bugs in edge color handling by contourf, found
           by Jae-Joon Lee. - EF

2009-03-14 Added 'LightSource' class to colors module for
           creating shaded relief maps.  shading_example.py
           added to illustrate usage. - JSW

2009-03-11 Ensure wx version >= 2.8; thanks to Sandro Tosi and
           Chris Barker. - EF

2009-03-10 Fix join style bug in pdf. - JKS

2009-03-07 Add pyplot access to figure number list - EF

2009-02-28 hashing of FontProperties accounts current rcParams - JJL

2009-02-28 Prevent double-rendering of shared axis in twinx, twiny - EF

2009-02-26 Add optional bbox_to_anchor argument for legend class - JJL

2009-02-26 Support image clipping in pdf backend. - JKS

2009-02-25 Improve tick location subset choice in FixedLocator. - EF

2009-02-24 Deprecate numerix, and strip out all but the numpy
           part of the code. - EF

2009-02-21 Improve scatter argument handling; add an early error
           message, allow inputs to have more than one dimension. - EF

2009-02-16 Move plot_directive.py to the installed source tree.  Add
           support for inline code content - MGD

2009-02-16 Move mathmpl.py to the installed source tree so it is
           available to other projects. - MGD

2009-02-14 Added the legend title support - JJL

2009-02-10 Fixed a bug in backend_pdf so it doesn't break when the setting
           pdf.use14corefonts=True is used. Added test case in
           unit/test_pdf_use14corefonts.py. - NGR

2009-02-08 Added a new imsave function to image.py and exposed it in
           the pyplot interface - GR

2009-02-04 Some reorgnization of the legend code. anchored_text.py
           added as an example. - JJL

2009-02-04 Add extent keyword arg to hexbin - ADS

2009-02-04 Fix bug in mathtext related to \dots and \ldots - MGD

2009-02-03 Change default joinstyle to round - MGD

2009-02-02 Reduce number of marker XObjects in pdf output - JKS

2009-02-02 Change default resolution on polar plot to 1 - MGD

2009-02-02 Avoid malloc errors in ttconv for fonts that don't have
           e.g., PostName (a version of Tahoma triggered this) - JKS

2009-01-30 Remove support for pyExcelerator in exceltools -- use xlwt
           instead - JDH

2009-01-29 Document 'resolution' kwarg for polar plots.  Support it
           when using pyplot.polar, not just Figure.add_axes. - MGD

2009-01-29 Rework the nan-handling/clipping/quantizing/simplification
           framework so each is an independent part of a pipeline.
           Expose the C++-implementation of all of this so it can be
           used from all Python backends.  Add rcParam
           "path.simplify_threshold" to control the threshold of
           similarity below which vertices will be removed.

2009-01-26 Improved tight bbox option of the savefig. - JJL

2009-01-26 Make curves and NaNs play nice together - MGD

2009-01-21 Changed the defaults of acorr and xcorr to use
           usevlines=True, maxlags=10 and normed=True since these are
           the best defaults

2009-01-19 Fix bug in quiver argument handling. - EF

2009-01-19 Fix bug in backend_gtk: don't delete nonexistent toolbar. - EF

2009-01-16 Implement bbox_inches option for savefig. If bbox_inches is
           "tight", try to determine the tight bounding box. - JJL

2009-01-16 Fix bug in is_string_like so it doesn't raise an
           unnecessary exception. - EF

2009-01-16 Fix an infinite recursion in the unit registry when searching
           for a converter for a sequence of strings. Add a corresponding
           test. - RM

2009-01-16 Bugfix of C typedef of MPL_Int64 that was failing on
           Windows XP 64 bit, as reported by George Goussard on numpy
           mailing list. - ADS

2009-01-16 Added helper function LinearSegmentedColormap.from_list to
           facilitate building simple custom colomaps.  See
           examples/pylab_examples/custom_cmap_fromlist.py - JDH

2009-01-16 Applied Michiel's patch for macosx backend to fix rounding
           bug. Closed sf bug 2508440 - JSW

2009-01-10 Applied Michiel's hatch patch for macosx backend and
           draw_idle patch for qt.  Closes sf patched 2497785 and
           2468809 - JDH

2009-01-10 Fix bug in pan/zoom with log coordinates. - EF

2009-01-06 Fix bug in setting of dashed negative contours. - EF

2009-01-06 Be fault tolerant when len(linestyles)>NLev in contour. - MM

2009-01-06 Added marginals kwarg to hexbin to plot marginal densities
           JDH

2009-01-06 Change user-visible multipage pdf object to PdfPages to
           avoid accidents with the file-like PdfFile. - JKS

2009-01-05 Fix a bug in pdf usetex: allow using non-embedded fonts. - JKS

2009-01-05 optional use of preview.sty in usetex mode. - JJL

2009-01-02 Allow multipage pdf files. - JKS

2008-12-31 Improve pdf usetex by adding support for font effects
           (slanting and extending). - JKS

2008-12-29 Fix a bug in pdf usetex support, which occurred if the same
           Type-1 font was used with different encodings, e.g., with
           Minion Pro and MnSymbol. - JKS

2008-12-20 fix the dpi-dependent offset of Shadow. - JJL

2008-12-20 fix the hatch bug in the pdf backend. minor update
           in docs and  example - JJL

2008-12-19 Add axes_locator attribute in Axes. Two examples are added.
           - JJL

2008-12-19 Update Axes.legend documnetation. /api/api_changes.rst is also
           updated to describe chages in keyword parameters.
           Issue a warning if old keyword parameters are used. - JJL

2008-12-18 add new arrow style, a line + filled triangles. -JJL

----------------

2008-12-18 Re-Released 0.98.5.2 from v0_98_5_maint at r6679
           Released 0.98.5.2 from v0_98_5_maint at r6667

2008-12-18 Removed configobj, experimental traits and doc/mpl_data link - JDH

2008-12-18 Fix bug where a line with NULL data limits prevents
           subsequent data limits from calculating correctly - MGD

2008-12-17 Major documentation generator changes - MGD

2008-12-17 Applied macosx backend patch with support for path
           collections, quadmesh, etc... - JDH

2008-12-17 fix dpi-dependent behavior of text bbox and arrow in annotate
            -JJL

2008-12-17 Add group id support in artist. Two examples which
           demostrate svg filter are added. -JJL

2008-12-16 Another attempt to fix dpi-dependent behavior of Legend. -JJL

2008-12-16 Fixed dpi-dependent behavior of Legend and fancybox in Text.

2008-12-16 Added markevery property to Line2D to support subsampling
           of markers - JDH
2008-12-15 Removed mpl_data symlink in docs.  On platforms that do not
           support symlinks, these become copies, and the font files
           are large, so the distro becomes unneccessarily bloaded.
           Keeping the mpl_examples dir because relative links are
           harder for the plot directive and the \*.py files are not so
           large. - JDH

2008-12-15 Fix \$ in non-math text with usetex off.  Document
           differences between usetex on/off - MGD

2008-12-15 Fix anti-aliasing when auto-snapping - MGD

2008-12-15 Fix grid lines not moving correctly during pan and zoom - MGD

2008-12-12 Preparations to eliminate maskedarray rcParams key: its
           use will now generate a warning.  Similarly, importing
           the obsolote numerix.npyma will generate a warning. - EF

2008-12-12 Added support for the numpy.histogram() weights parameter
           to the axes hist() method. Docs taken from numpy - MM

2008-12-12 Fixed warning in hist() with numpy 1.2 - MM

2008-12-12 Removed external packages: configobj and enthought.traits
           which are only required by the experimental traited config
           and are somewhat out of date. If needed, install them
           independently, see:

           http://code.enthought.com/pages/traits.html

           and:

           http://www.voidspace.org.uk/python/configobj.html

2008-12-12 Added support to asign labels to histograms of multiple
           data. - MM

-------------------------

2008-12-11 Released 0.98.5 at svn r6573

2008-12-11 Use subprocess.Popen instead of os.popen in dviread
           (Windows problem reported by Jorgen Stenarson) - JKS

2008-12-10 Added Michael's font_manager fix and Jae-Joon's
           figure/subplot fix.  Bumped version number to 0.98.5 - JDH

----------------------------

2008-12-09 Released 0.98.4 at svn r6536

2008-12-08 Added mdehoon's native macosx backend from sf patch 2179017 - JDH

2008-12-08 Removed the prints in the set_*style commands.  Return the
           list of pprinted strings instead - JDH

2008-12-08 Some of the changes Michael made to improve the output of
           the property tables in the rest docs broke of made
           difficult to use some of the interactive doc helpers, e.g.,
           setp and getp.  Having all the rest markup in the ipython
           shell also confused the docstrings.  I added a new rc param
           docstring.harcopy, to format the docstrings differently for
           hardcopy and other use.  Ther ArtistInspector could use a
           little refactoring now since there is duplication of effort
           between the rest out put and the non-rest output - JDH

2008-12-08 Updated spectral methods (psd, csd, etc.) to scale one-sided
           densities by a factor of 2 and, optionally, scale all densities
           by the sampling frequency.  This gives better MatLab
           compatibility. -RM

2008-12-08 Fixed alignment of ticks in colorbars. -MGD

2008-12-07 drop the deprecated "new" keyword of np.histogram() for
           numpy 1.2 or later.  -JJL

2008-12-06 Fixed a bug in svg backend that new_figure_manager()
           ignores keywords arguments such as figsize, etc. -JJL

2008-12-05 Fixed a bug that the handlelength of the new legend class
           set too short when numpoints=1 -JJL

2008-12-04 Added support for data with units (e.g., dates) to
           Axes.fill_between. -RM

2008-12-04 Added fancybox keyword to legend. Also applied some changes
           for better look, including baseline adjustment of the
           multiline texts so that it is center aligned. -JJL

2008-12-02 The transmuter classes in the patches.py are reorganized as
           subclasses of the Style classes. A few more box and arrow
           styles are added. -JJL

2008-12-02 Fixed a bug in the new legend class that didn't allowed
           a tuple of coordinate vlaues as loc. -JJL

2008-12-02 Improve checks for external dependencies, using subprocess
           (instead of deprecated popen*) and distutils (for version
           checking) - DSD

2008-11-30 Reimplementation of the legend which supports baseline alignement,
           multi-column, and expand mode. - JJL

2008-12-01 Fixed histogram autoscaling bug when bins or range are given
           explicitly (fixes Debian bug 503148) - MM

2008-11-25 Added rcParam axes.unicode_minus which allows plain hypen
           for minus when False - JDH

2008-11-25 Added scatterpoints support in Legend. patch by Erik
           Tollerud - JJL

2008-11-24 Fix crash in log ticking. - MGD

2008-11-20 Added static helper method BrokenHBarCollection.span_where
           and Axes/pyplot method fill_between.  See
           examples/pylab/fill_between.py - JDH

2008-11-12 Add x_isdata and y_isdata attributes to Artist instances,
           and use them to determine whether either or both
           coordinates are used when updating dataLim.  This is
           used to fix autoscaling problems that had been triggered
           by axhline, axhspan, axvline, axvspan. - EF

2008-11-11 Update the psd(), csd(), cohere(), and specgram() methods
           of Axes and the csd() cohere(), and specgram() functions
           in mlab to be in sync with the changes to psd().
           In fact, under the hood, these all call the same core
           to do computations. - RM

2008-11-11 Add 'pad_to' and 'sides' parameters to mlab.psd() to
           allow controlling of zero padding and returning of
           negative frequency components, respecitively.  These are
           added in a way that does not change the API. - RM

2008-11-10 Fix handling of c kwarg by scatter; generalize
           is_string_like to accept numpy and numpy.ma string
           array scalars. - RM and EF

2008-11-09 Fix a possible EINTR problem in dviread, which might help
           when saving pdf files from the qt backend. - JKS

2008-11-05 Fix bug with zoom to rectangle and twin axes - MGD

2008-10-24 Added Jae Joon's fancy arrow, box and annotation
           enhancements -- see
           examples/pylab_examples/annotation_demo2.py

2008-10-23 Autoscaling is now supported with shared axes - EF

2008-10-23 Fixed exception in dviread that happened with Minion - JKS

2008-10-21 set_xlim, ylim now return a copy of the viewlim array to
           avoid modify inplace surprises

2008-10-20 Added image thumbnail generating function
           matplotlib.image.thumbnail.  See
           examples/misc/image_thumbnail.py - JDH

2008-10-20 Applied scatleg patch based on ideas and work by Erik
           Tollerud and Jae-Joon Lee. - MM

2008-10-11 Fixed bug in pdf backend: if you pass a file object for
           output instead of a filename, e.g., in a wep app, we now
           flush the object at the end. - JKS

2008-10-08 Add path simplification support to paths with gaps. - EF

2008-10-05 Fix problem with AFM files that don't specify the font's
           full name or family name. - JKS

2008-10-04 Added 'scilimits' kwarg to Axes.ticklabel_format() method,
           for easy access to the set_powerlimits method of the
           major ScalarFormatter. - EF

2008-10-04 Experimental new kwarg borderpad to replace pad in legend,
           based on suggestion by Jae-Joon Lee.  - EF

2008-09-27 Allow spy to ignore zero values in sparse arrays, based
           on patch by Tony Yu.  Also fixed plot to handle empty
           data arrays, and fixed handling of markers in figlegend. - EF

2008-09-24 Introduce drawstyles for lines. Transparently split linestyles
           like 'steps--' into drawstyle 'steps' and linestyle '--'.
           Legends always use drawstyle 'default'. - MM

2008-09-18 Fixed quiver and quiverkey bugs (failure to scale properly
           when resizing) and added additional methods for determining
           the arrow angles - EF

2008-09-18 Fix polar interpolation to handle negative values of theta - MGD

2008-09-14 Reorganized cbook and mlab methods related to numerical
           calculations that have little to do with the goals of those two
           modules into a separate module numerical_methods.py
           Also, added ability to select points and stop point selection
           with keyboard in ginput and manual contour labeling code.
           Finally, fixed contour labeling bug. - DMK

2008-09-11 Fix backtick in Postscript output. - MGD

2008-09-10 [ 2089958 ] Path simplification for vector output backends
           Leverage the simplification code exposed through
           path_to_polygons to simplify certain well-behaved paths in
           the vector backends (PDF, PS and SVG).  "path.simplify"
           must be set to True in matplotlibrc for this to work.  -
           MGD

2008-09-10 Add "filled" kwarg to Path.intersects_path and
           Path.intersects_bbox. - MGD

2008-09-07 Changed full arrows slightly to avoid an xpdf rendering
           problem reported by Friedrich Hagedorn. - JKS

2008-09-07 Fix conversion of quadratic to cubic Bezier curves in PDF
           and PS backends. Patch by Jae-Joon Lee. - JKS

2008-09-06 Added 5-point star marker to plot command - EF

2008-09-05 Fix hatching in PS backend - MGD

2008-09-03 Fix log with base 2 - MGD

2008-09-01 Added support for bilinear interpolation in
           NonUniformImage; patch by Gregory Lielens. - EF

2008-08-28 Added support for multiple histograms with data of
           different length - MM

2008-08-28 Fix step plots with log scale - MGD

2008-08-28 Fix masked arrays with markers in non-Agg backends - MGD

2008-08-28 Fix clip_on kwarg so it actually works correctly - MGD

2008-08-25 Fix locale problems in SVG backend - MGD

2008-08-22 fix quiver so masked values are not plotted - JSW

2008-08-18 improve interactive pan/zoom in qt4 backend on windows - DSD

2008-08-11 Fix more bugs in NaN/inf handling.  In particular, path simplification
           (which does not handle NaNs or infs) will be turned off automatically
           when infs or NaNs are present.  Also masked arrays are now converted
           to arrays with NaNs for consistent handling of masks and NaNs
           - MGD and EF

------------------------

2008-08-03 Released 0.98.3 at svn r5947

2008-08-01 Backported memory leak fixes in _ttconv.cpp - MGD

2008-07-31 Added masked array support to griddata. - JSW

2008-07-26 Added optional C and reduce_C_function arguments to
           axes.hexbin().  This allows hexbin to accumulate the values
           of C based on the x,y coordinates and display in hexagonal
           bins. - ADS

2008-07-24 Deprecated (raise NotImplementedError) all the mlab2
           functions from matplotlib.mlab out of concern that some of
           them were not clean room implementations. JDH

2008-07-24 Rewrite of a significant portion of the clabel code (class
           ContourLabeler) to improve inlining. - DMK

2008-07-22 Added Barbs polygon collection (similar to Quiver) for plotting
           wind barbs.  Added corresponding helpers to Axes and pyplot as
           well. (examples/pylab_examples/barb_demo.py shows it off.) - RMM

2008-07-21 Added scikits.delaunay as matplotlib.delaunay.  Added griddata
           function in matplotlib.mlab, with example (griddata_demo.py) in
           pylab_examples. griddata function will use mpl_toolkits._natgrid
           if installed.  - JSW

2008-07-21 Re-introduced offset_copy that works in the context of the
           new transforms. - MGD

2008-07-21 Committed patch by Ryan May to add get_offsets and
           set_offsets to Collections base class - EF

2008-07-21 Changed the "asarray" strategy in image.py so that
           colormapping of masked input should work for all
           image types (thanks Klaus Zimmerman) - EF

2008-07-20 Rewrote cbook.delete_masked_points and corresponding
           unit test to support rgb color array inputs, datetime
           inputs, etc. - EF

2008-07-20 Renamed unit/axes_unit.py to cbook_unit.py and modified
           in accord with Ryan's move of delete_masked_points from
           axes to cbook. - EF

2008-07-18 Check for nan and inf in axes.delete_masked_points().
           This should help hexbin and scatter deal with nans. - ADS

2008-07-17 Added ability to manually select contour label locations.
           Also added a waitforbuttonpress function. - DMK

2008-07-17 Fix bug with NaNs at end of path (thanks, Andrew Straw for
           the report) - MGD

2008-07-16 Improve error handling in texmanager, thanks to Ian Henry
           for reporting - DSD

2008-07-12 Added support for external backends with the
           "module://my_backend" syntax - JDH

2008-07-11 Fix memory leak related to shared axes.  Grouper should
           store weak references. - MGD

2008-07-10 Bugfix: crash displaying fontconfig pattern - MGD

2008-07-10 Bugfix: [ 2013963 ] update_datalim_bounds in Axes not works - MGD

2008-07-10 Bugfix: [ 2014183 ] multiple imshow() causes gray edges - MGD

2008-07-09 Fix rectangular axes patch on polar plots bug - MGD

2008-07-09 Improve mathtext radical rendering - MGD

2008-07-08 Improve mathtext superscript placement - MGD

2008-07-07 Fix custom scales in pcolormesh (thanks Matthew Turk) - MGD

2008-07-03 Implemented findobj method for artist and pyplot - see
           examples/pylab_examples/findobj_demo.py - JDH

2008-06-30 Another attempt to fix TextWithDash - DSD

2008-06-30 Removed Qt4 NavigationToolbar2.destroy -- it appears to
           have been unnecessary and caused a bug reported by P.
           Raybaut - DSD

2008-06-27 Fixed tick positioning bug - MM

2008-06-27 Fix dashed text bug where text was at the wrong end of the
           dash - MGD

2008-06-26 Fix mathtext bug for expressions like $x_{\leftarrow}$ - MGD

2008-06-26 Fix direction of horizontal/vertical hatches - MGD

2008-06-25 Figure.figurePatch renamed Figure.patch, Axes.axesPatch
           renamed Axes.patch, Axes.axesFrame renamed Axes.frame,
           Axes.get_frame, which returns Axes.patch, is deprecated.
           Examples and users guide updated - JDH

2008-06-25 Fix rendering quality of pcolor - MGD

----------------------------

2008-06-24 Released 0.98.2 at svn r5667 - (source only for debian) JDH

2008-06-24 Added "transparent" kwarg to savefig. - MGD

2008-06-24 Applied Stefan's patch to draw a single centered marker over
           a line with numpoints==1 - JDH

2008-06-23 Use splines to render circles in scatter plots - MGD

----------------------------

2008-06-22 Released 0.98.1 at revision 5637

2008-06-22 Removed axes3d support and replaced it with a
           NotImplementedError for one release cycle

2008-06-21 fix marker placement bug in backend_ps - DSD

2008-06-20 [ 1978629 ] scale documentation missing/incorrect for log - MGD

2008-06-20 Added closed kwarg to PolyCollection.  Fixes bug [ 1994535
           ] still missing lines on graph with svn (r 5548). - MGD

2008-06-20 Added set/get_closed method to Polygon; fixes error
           in hist - MM

2008-06-19 Use relative font sizes (e.g., 'medium' and 'large') in
           rcsetup.py and matplotlibrc.template so that text will
           be scaled by default when changing rcParams['font.size'] -
           EF

2008-06-17 Add a generic PatchCollection class that can contain any
           kind of patch. - MGD

2008-06-13 Change pie chart label alignment to avoid having labels
           overwrite the pie - MGD

2008-06-12 Added some helper functions to the mathtext parser to
           return bitmap arrays or write pngs to make it easier to use
           mathtext outside the context of an mpl figure.  modified
           the mathpng sphinxext to use the mathtext png save
           functionality - see examples/api/mathtext_asarray.py - JDH

2008-06-11 Use matplotlib.mathtext to render math expressions in
           online docs - MGD

2008-06-11 Move PNG loading/saving to its own extension module, and
           remove duplicate code in _backend_agg.cpp and _image.cpp
           that does the same thing - MGD

2008-06-11 Numerous mathtext bugfixes, primarily related to
           dpi-independence - MGD

2008-06-10 Bar now applies the label only to the first patch only, and
           sets '_nolegend_' for the other patch labels.  This lets
           autolegend work as expected for hist and bar - see
           https://sourceforge.net/tracker/index.php?func=detail&aid=1986597&group_id=80706&atid=560720
           JDH

2008-06-10 Fix text baseline alignment bug.  [ 1985420 ] Repair of
           baseline alignment in Text._get_layout.  Thanks Stan West -
           MGD

2008-06-09 Committed Gregor's image resample patch to downsampling
           images with new rcparam image.resample - JDH

2008-06-09 Don't install Enthought.Traits along with matplotlib. For
           matplotlib developers convenience, it can still be
           installed by setting an option in setup.cfg while we figure
           decide if there is a future for the traited config - DSD

2008-06-09 Added range keyword arg to hist() - MM

2008-06-07 Moved list of backends to rcsetup.py; made use of lower
           case for backend names consistent; use validate_backend
           when importing backends subpackage - EF

2008-06-06 hist() revision, applied ideas proposed by Erik Tollerud and
           Olle Engdegard: make histtype='step' unfilled by default
           and introduce histtype='stepfilled'; use default color
           cycle; introduce reverse cumulative histogram; new align
           keyword - MM

2008-06-06 Fix closed polygon patch and also provide the option to
           not close the polygon - MGD

2008-06-05 Fix some dpi-changing-related problems with PolyCollection,
           as called by Axes.scatter() - MGD

2008-06-05 Fix image drawing so there is no extra space to the right
           or bottom - MGD

2006-06-04 Added a figure title command suptitle as a Figure method
           and pyplot command -- see examples/figure_title.py - JDH

2008-06-02 Added support for log to hist with histtype='step' and fixed
           a bug for log-scale stacked histograms - MM

-----------------------------

2008-05-29 Released 0.98.0 at revision 5314

2008-05-29 matplotlib.image.imread now no longer always returns RGBA
           -- if the image is luminance or RGB, it will return a MxN
           or MxNx3 array if possible.  Also uint8 is no longer always
           forced to float.

2008-05-29 Implement path clipping in PS backend - JDH

2008-05-29 Fixed two bugs in texmanager.py:
           improved comparison of dvipng versions
           fixed a bug introduced when get_grey method was added
           - DSD

2008-05-28 Fix crashing of PDFs in xpdf and ghostscript when two-byte
           characters are used with Type 3 fonts - MGD

2008-05-28 Allow keyword args to configure widget properties as
           requested in
           http://sourceforge.net/tracker/index.php?func=detail&aid=1866207&group_id=80706&atid=560722
           - JDH

2008-05-28 Replaced '-' with u'\u2212' for minus sign as requested in
           http://sourceforge.net/tracker/index.php?func=detail&aid=1962574&group_id=80706&atid=560720

2008-05-28 zero width/height Rectangles no longer influence the
           autoscaler.  Useful for log histograms with empty bins -
           JDH

2008-05-28 Fix rendering of composite glyphs in Type 3 conversion
           (particularly as evidenced in the Eunjin.ttf Korean font)
           Thanks Jae-Joon Lee for finding this!

2008-05-27 Rewrote the cm.ScalarMappable callback infrastructure to
           use cbook.CallbackRegistry rather than custom callback
           handling.  Amy users of add_observer/notify of the
           cm.ScalarMappable should uae the
           cm.ScalarMappable.callbacksSM CallbackRegistry instead. JDH

2008-05-27 Fix TkAgg build on Ubuntu 8.04 (and hopefully a more
           general solution for other platforms, too.)

2008-05-24 Added PIL support for loading images to imread (if PIL is
           available) - JDH

2008-05-23 Provided a function and a method for controlling the
           plot color cycle. - EF

2008-05-23 Major revision of hist(). Can handle 2D arrays and create
           stacked histogram plots; keyword 'width' deprecated and
           rwidth (relative width) introduced; align='edge' changed
           to center of bin - MM

2008-05-22 Added support for ReST-based doumentation using Sphinx.
           Documents are located in doc/, and are broken up into
           a users guide and an API reference. To build, run the
           make.py files. Sphinx-0.4 is needed to build generate xml,
           which will be useful for rendering equations with mathml,
           use sphinx from svn until 0.4 is released - DSD

2008-05-21 Fix segfault in TkAgg backend - MGD

2008-05-21 Fix a "local variable unreferenced" bug in plotfile - MM

2008-05-19 Fix crash when Windows can not access the registry to
           determine font path [Bug 1966974, thanks Patrik Simons] - MGD

2008-05-16 removed some unneeded code w/ the python 2.4 requirement.
           cbook no longer provides compatibility for reversed,
           enumerate, set or izip.  removed lib/subprocess, mpl1,
           sandbox/units, and the swig code.  This stuff should remain
           on the maintenance branch for archival purposes. JDH

2008-05-16 Reorganized examples dir - JDH

2008-05-16 Added 'elinewidth' keyword arg to errorbar, based on patch
           by Christopher Brown - MM

2008-05-16 Added 'cumulative' keyword arg to hist to plot cumulative
           histograms. For normed hists, this is normalized to one - MM

2008-05-15 Fix Tk backend segfault on some machines - MGD

2008-05-14 Don't use stat on Windows (fixes font embedding problem) - MGD

2008-05-09 Fix /singlequote (') in Postscript backend - MGD

2008-05-08 Fix kerning in SVG when embedding character outlines - MGD

2008-05-07 Switched to future numpy histogram semantic in hist - MM

2008-05-06 Fix strange colors when blitting in QtAgg and Qt4Agg - MGD

2008-05-05 pass notify_axes_change to the figure's add_axobserver
           in the qt backends, like we do for the other backends.
           Thanks Glenn Jones for the report - DSD

2008-05-02 Added step histograms, based on patch by Erik Tollerud. - MM

2008-05-02 On PyQt <= 3.14 there is no way to determine the underlying
           Qt version. [1851364] - MGD

2008-05-02 Don't call sys.exit() when pyemf is not found [1924199] -
           MGD

2008-05-02 Update _subprocess.c from upstream Python 2.5.2 to get a
           few memory and reference-counting-related bugfixes.  See
           bug 1949978. - MGD

2008-04-30 Added some record array editing widgets for gtk -- see
           examples/rec_edit*.py - JDH

2008-04-29 Fix bug in mlab.sqrtm - MM

2008-04-28 Fix bug in SVG text with Mozilla-based viewers (the symbol
           tag is not supported) - MGD

2008-04-27 Applied patch by Michiel de Hoon to add hexbin
           axes method and pyplot function - EF

2008-04-25 Enforce python >= 2.4; remove subprocess build - EF

2008-04-25 Enforce the numpy requirement at build time - JDH

2008-04-24 Make numpy 1.1 and python 2.3 required when importing
           matplotlib - EF

2008-04-24 Fix compilation issues on VS2003 (Thanks Martin Spacek for
           all the help) - MGD

2008-04-24 Fix sub/superscripts when the size of the font has been
           changed - MGD

2008-04-22 Use "svg.embed_char_paths" consistently everywhere - MGD

2008-04-20 Add support to MaxNLocator for symmetric axis autoscaling. - EF

2008-04-20 Fix double-zoom bug. - MM

2008-04-15 Speed up color mapping. - EF

2008-04-12 Speed up zooming and panning of dense images. - EF

2008-04-11 Fix global font rcParam setting after initialization
           time. - MGD

2008-04-11 Revert commits 5002 and 5031, which were intended to
           avoid an unnecessary call to draw(). 5002 broke saving
           figures before show(). 5031 fixed the problem created in
           5002, but broke interactive plotting. Unnecessary call to
           draw still needs resolution - DSD

2008-04-07 Improve color validation in rc handling, suggested
           by Lev Givon - EF

2008-04-02 Allow to use both linestyle definition arguments, '-' and
           'solid' etc. in plots/collections - MM

2008-03-27 Fix saving to Unicode filenames with Agg backend
           (other backends appear to already work...)
           (Thanks, Christopher Barker) - MGD

2008-03-26 Fix SVG backend bug that prevents copying and pasting in
           Inkscape (thanks Kaushik Ghose) - MGD

2008-03-24 Removed an unnecessary call to draw() in the backend_qt*
           mouseReleaseEvent. Thanks to Ted Drain - DSD

2008-03-23 Fix a pdf backend bug which sometimes caused the outermost
           gsave to not be balanced with a grestore. - JKS

2008-03-20 Fixed a minor bug in ContourSet._process_linestyles when
           len(linestyles)==Nlev - MM

2008-03-19 Changed ma import statements to "from numpy import ma";
           this should work with past and future versions of
           numpy, whereas "import numpy.ma as ma" will work only
           with numpy >= 1.05, and "import numerix.npyma as ma"
           is obsolete now that maskedarray is replacing the
           earlier implementation, as of numpy 1.05.

2008-03-14 Removed an apparently unnecessary call to
           FigureCanvasAgg.draw in backend_qt*agg. Thanks to Ted
           Drain - DSD

2008-03-10 Workaround a bug in backend_qt4agg's blitting due to a
           buffer width/bbox width mismatch in _backend_agg's
           copy_from_bbox - DSD

2008-02-29 Fix class Wx toolbar pan and zoom functions (Thanks Jeff
           Peery) - MGD

2008-02-16 Added some new rec array functionality to mlab
           (rec_summarize, rec2txt and rec_groupby).  See
           examples/rec_groupby_demo.py.  Thanks to Tim M for rec2txt.

2008-02-12 Applied Erik Tollerud's span selector patch - JDH

2008-02-11 Update plotting() doc string to refer to getp/setp. - JKS

2008-02-10 Fixed a problem with square roots in the pdf backend with
           usetex. - JKS

2008-02-08 Fixed minor __str__ bugs so getp(gca()) works. - JKS

2008-02-05 Added getters for title, xlabel, ylabel, as requested
           by Brandon Kieth - EF

2008-02-05 Applied Gael's ginput patch and created
           examples/ginput_demo.py - JDH

2008-02-03 Expose interpnames, a list of valid interpolation
           methods, as an AxesImage class attribute. - EF

2008-02-03 Added BoundaryNorm, with examples in colorbar_only.py
           and image_masked.py. - EF

2008-02-03 Force dpi=72 in pdf backend to fix picture size bug. - JKS

2008-02-01 Fix doubly-included font problem in Postscript backend - MGD

2008-02-01 Fix reference leak in ft2font Glyph objects. - MGD

2008-01-31 Don't use unicode strings with usetex by default - DSD

2008-01-31 Fix text spacing problems in PDF backend with *some* fonts,
           such as STIXGeneral.

2008-01-31 Fix \sqrt with radical number (broken by making [ and ]
           work below) - MGD

2008-01-27 Applied  Martin Teichmann's patch to improve the Qt4
           backend. Uses Qt's builtin toolbars and statusbars.
           See bug 1828848 - DSD

2008-01-10 Moved toolkits to mpl_toolkits, made mpl_toolkits
           a namespace package - JSWHIT

2008-01-10 Use setup.cfg to set the default parameters (tkagg,
           numpy) when building windows installers - DSD

2008-01-10 Fix bug displaying [ and ] in mathtext - MGD

2008-01-10 Fix bug when displaying a tick value offset with scientific
           notation. (Manifests itself as a warning that the \times
           symbol can not be found). - MGD

2008-01-10 Use setup.cfg to set the default parameters (tkagg,
           numpy) when building windows installers - DSD

--------------------

2008-01-06 Released 0.91.2 at revision 4802

2007-12-26 Reduce too-late use of matplotlib.use() to a warning
           instead of an exception, for backwards compatibility - EF

2007-12-25 Fix bug in errorbar, identified by Noriko Minakawa - EF

2007-12-25 Changed masked array importing to work with the upcoming
           numpy 1.05 (now the maskedarray branch) as well as with
           earlier versions. - EF

2007-12-16 rec2csv saves doubles without losing precision. Also, it
           does not close filehandles passed in open. - JDH,ADS

2007-12-13 Moved rec2gtk to matplotlib.toolkits.gtktools and rec2excel
           to matplotlib.toolkits.exceltools - JDH

2007-12-12 Support alpha-blended text in the Agg and Svg backends -
           MGD

2007-12-10 Fix SVG text rendering bug. - MGD

2007-12-10 Increase accuracy of circle and ellipse drawing by using an
           8-piece bezier approximation, rather than a 4-piece one.
           Fix PDF, SVG and Cairo backends so they can draw paths
           (meaning ellipses as well). - MGD

2007-12-07 Issue a warning when drawing an image on a non-linear axis. - MGD

2007-12-06 let widgets.Cursor initialize to the lower x and y bounds
           rather than 0,0, which can cause havoc for dates and other
           transforms - DSD

2007-12-06 updated references to mpl data directories for py2exe - DSD

2007-12-06 fixed a bug in rcsetup, see bug 1845057 - DSD

2007-12-05 Fix how fonts are cached to avoid loading the same one multiple times.
           (This was a regression since 0.90 caused by the refactoring of
           font_manager.py) - MGD

2007-12-05 Support arbitrary rotation of usetex text in Agg backend. - MGD

2007-12-04 Support '|' as a character in mathtext - MGD

-----------------------------------------------------

2007-11-27 Released 0.91.1 at revision 4517

-----------------------------------------------------

2007-11-27 Released 0.91.0 at revision 4478

2007-11-13 All backends now support writing to a file-like object, not
           just a regular file.  savefig() can be passed a file-like
           object in place of a file path. - MGD

2007-11-13 Improved the default backend selection at build time:
           SVG -> Agg -> TkAgg -> WXAgg -> GTK -> GTKAgg. The last usable
           backend in this progression will be chosen in the default
           config file. If a backend is defined in setup.cfg, that will
           be the default backend - DSD

2007-11-13 Improved creation of default config files at build time for
           traited config package - DSD

2007-11-12 Exposed all the build options in setup.cfg. These options are
           read into a dict called "options" by setupext.py. Also, added
           "-mpl" tags to the version strings for packages provided by
           matplotlib. Versions provided by mpl will be identified and
           updated on subsequent installs - DSD

2007-11-12 Added support for STIX fonts.  A new rcParam,
           mathtext.fontset, can be used to choose between:

           'cm':
             The TeX/LaTeX Computer Modern fonts

           'stix':
             The STIX fonts (see stixfonts.org)

           'stixsans':
             The STIX fonts, using sans-serif glyphs by default

           'custom':
             A generic Unicode font, in which case the mathtext font
             must be specified using mathtext.bf, mathtext.it,
             mathtext.sf etc.

           Added a new example, stix_fonts_demo.py to show how to access
           different fonts and unusual symbols.

           - MGD

2007-11-12 Options to disable building backend extension modules moved
           from setup.py to setup.cfg - DSD

2007-11-09 Applied Martin Teichmann's patch 1828813: a QPainter is used in
           paintEvent, which has to be destroyed using  the method end(). If
           matplotlib raises an exception before the call to end - and it
           does if you feed it with bad data - this method end() is never
           called and Qt4 will start spitting error messages

2007-11-09 Moved pyparsing back into matplotlib namespace. Don't use
           system pyparsing, API is too variable from one release
           to the next - DSD

2007-11-08 Made pylab use straight numpy instead of oldnumeric
           by default - EF

2007-11-08 Added additional record array utilites to mlab (rec2excel,
           rec2gtk, rec_join, rec_append_field, rec_drop_field) - JDH

2007-11-08 Updated pytz to version 2007g - DSD

2007-11-08 Updated pyparsing to version 1.4.8 - DSD

2007-11-08 Moved csv2rec to recutils and added other record array
           utilities - JDH

2007-11-08 If available, use existing pyparsing installation - DSD

2007-11-07 Removed old enthought.traits from lib/matplotlib, added
           Gael Varoquaux's enthought.traits-2.6b1, which is stripped
           of setuptools. The package is installed to site-packages
           if not already available - DSD

2007-11-05 Added easy access to minor tick properties; slight mod
           of patch by Pierre G-M - EF

2007-11-02 Commited Phil Thompson's patch 1599876, fixes to Qt4Agg
           backend and qt4 blitting demo - DSD

2007-11-02 Commited Phil Thompson's patch 1599876, fixes to Qt4Agg
           backend and qt4 blitting demo - DSD

2007-10-31 Made log color scale easier to use with contourf;
           automatic level generation now works. - EF

2007-10-29 TRANSFORMS REFACTORING

           The primary goal of this refactoring was to make it easier
           to extend matplotlib to support new kinds of projections.
           This is primarily an internal improvement, and the possible
           user-visible changes it allows are yet to come.

           The transformation framework was completely rewritten in
           Python (with Numpy).  This will make it easier to add news
           kinds of transformations without writing C/C++ code.

           Transforms are composed into a 'transform tree', made of
           transforms whose value depends on other transforms (their
           children).  When the contents of children change, their
           parents are automatically updated to reflect those changes.
           To do this an "invalidation" method is used: when children
           change, all of their ancestors are marked as "invalid".
           When the value of a transform is accessed at a later time,
           its value is recomputed only if it is invalid, otherwise a
           cached value may be used.  This prevents unnecessary
           recomputations of transforms, and contributes to better
           interactive performance.

           The framework can be used for both affine and non-affine
           transformations.  However, for speed, we want use the
           backend renderers to perform affine transformations
           whenever possible.  Therefore, it is possible to perform
           just the affine or non-affine part of a transformation on a
           set of data.  The affine is always assumed to occur after
           the non-affine.  For any transform:

                full transform == non-affine + affine

           Much of the drawing has been refactored in terms of
           compound paths.  Therefore, many methods have been removed
           from the backend interface and replaced with a a handful to
           draw compound paths.  This will make updating the backends
           easier, since there is less to update.  It also should make
           the backends more consistent in terms of functionality.

           User visible changes:

           - POLAR PLOTS: Polar plots are now interactively zoomable,
             and the r-axis labels can be interactively rotated.
             Straight line segments are now interpolated to follow the
             curve of the r-axis.

           - Non-rectangular clipping works in more backends and with
             more types of objects.

           - Sharing an axis across figures is now done in exactly
             the same way as sharing an axis between two axes in the
             same figure::

                  fig1 = figure()
                  fig2 = figure()

                   ax1 = fig1.add_subplot(111)
                   ax2 = fig2.add_subplot(111, sharex=ax1, sharey=ax1)

           - linestyles now include steps-pre, steps-post and
             steps-mid.  The old step still works and is equivalent to
             step-pre.

           - Multiple line styles may be provided to a collection.

           See API_CHANGES for more low-level information about this
           refactoring.

2007-10-24 Added ax kwarg to Figure.colorbar and pyplot.colorbar - EF

2007-10-19 Removed a gsave/grestore pair surrounding _draw_ps, which
           was causing a loss graphics state info (see "EPS output
           problem - scatter & edgecolors" on mpl-dev, 2007-10-29)
           - DSD

2007-10-15 Fixed a bug in patches.Ellipse that was broken for
           aspect='auto'.  Scale free ellipses now work properly for
           equal and auto on Agg and PS, and they fall back on a
           polygonal approximation for nonlinear transformations until
           we convince oursleves that the spline approximation holds
           for nonlinear transformations. Added
           unit/ellipse_compare.py to compare spline with vertex
           approx for both aspects. JDH

2007-10-05 remove generator expressions from texmanager and mpltraits.
           generator expressions are not supported by python-2.3 - DSD

2007-10-01 Made matplotlib.use() raise an exception if called after
           backends has been imported. - EF

2007-09-30 Modified update* methods of Bbox and Interval so they
           work with reversed axes.  Prior to this, trying to
           set the ticks on a reversed axis failed with an
           uninformative error message. - EF

2007-09-30 Applied patches to axes3d to fix index error problem - EF

2007-09-24 Applied Eike Welk's patch reported on mpl-dev on 2007-09-22
           Fixes a bug with multiple plot windows in the qt backend,
           ported the changes to backend_qt4 as well - DSD

2007-09-21 Changed cbook.reversed to yield the same result as the
           python reversed builtin - DSD

2007-09-13 The usetex support in the pdf backend is more usable now,
           so I am enabling it. - JKS

2007-09-12 Fixed a Axes.bar unit bug - JDH

2007-09-10 Made skiprows=1 the default on csv2rec - JDH

2007-09-09 Split out the plotting part of pylab and put it in
           pyplot.py; removed numerix from the remaining pylab.py,
           which imports everything from pyplot.py.  The intention
           is that apart from cleanups, the result of importing
           from pylab is nearly unchanged, but there is the
           new alternative of importing from pyplot to get
           the state-engine graphics without all the numeric
           functions.
           Numpified examples; deleted two that were obsolete;
           modified some to use pyplot. - EF

2007-09-08 Eliminated gd and paint backends - EF

2007-09-06 .bmp file format is now longer an alias for .raw

2007-09-07 Added clip path support to pdf backend. - JKS

2007-09-06 Fixed a bug in the embedding of Type 1 fonts in PDF.
           Now it doesn't crash Preview.app. - JKS

2007-09-06 Refactored image saving code so that all GUI backends can
           save most image types.  See FILETYPES for a matrix of
           backends and their supported file types.
           Backend canvases should no longer write their own print_figure()
           method -- instead they should write a print_xxx method for
           each filetype they can output and add an entry to their
           class-scoped filetypes dictionary. - MGD

2007-09-05 Fixed Qt version reporting in setupext.py - DSD

2007-09-04 Embedding Type 1 fonts in PDF, and thus usetex support
           via dviread, sort of works. To test, enable it by
           renaming _draw_tex to draw_tex. - JKS

2007-09-03 Added ability of errorbar show limits via caret or
           arrowhead ends on the bars; patch by Manual Metz. - EF

2007-09-03 Created type1font.py, added features to AFM and FT2Font
           (see API_CHANGES), started work on embedding Type 1 fonts
           in pdf files. - JKS

2007-09-02 Continued work on dviread.py. - JKS

2007-08-16 Added a set_extent method to AxesImage, allow data extent
           to be modified after initial call to imshow - DSD

2007-08-14 Fixed a bug in pyqt4 subplots-adjust. Thanks to
           Xavier Gnata for the report and suggested fix - DSD

2007-08-13 Use pickle to cache entire fontManager; change to using
            font_manager module-level function findfont wrapper for
            the fontManager.findfont method - EF

2007-08-11 Numpification and cleanup of mlab.py and some examples - EF

2007-08-06 Removed mathtext2

2007-07-31 Refactoring of distutils scripts.
           - Will not fail on the entire build if an optional Python
             package (e.g., Tkinter) is installed but its development
             headers are not (e.g., tk-devel).  Instead, it will
             continue to build all other extensions.
           - Provide an overview at the top of the output to display
             what dependencies and their versions were found, and (by
             extension) what will be built.
           - Use pkg-config, when available, to find freetype2, since
             this was broken on Mac OS-X when using MacPorts in a non-
             standard location.

2007-07-30 Reorganized configuration code to work with traited config
           objects. The new config system is located in the
           matplotlib.config package, but it is disabled by default.
           To enable it, set NEWCONFIG=True in matplotlib.__init__.py.
           The new configuration system will still use the old
           matplotlibrc files by default. To switch to the experimental,
           traited configuration, set USE_TRAITED_CONFIG=True in
           config.__init__.py.

2007-07-29 Changed default pcolor shading to flat; added aliases
           to make collection kwargs agree with setter names, so
           updating works; related minor cleanups.
           Removed quiver_classic, scatter_classic, pcolor_classic. - EF

2007-07-26 Major rewrite of mathtext.py, using the TeX box layout model.

           There is one (known) backward incompatible change.  The
           font commands (\cal, \rm, \it, \tt) now behave as TeX does:
           they are in effect until the next font change command or
           the end of the grouping.  Therefore uses of $\cal{R}$
           should be changed to ${\cal R}$.  Alternatively, you may
           use the new LaTeX-style font commands (\mathcal, \mathrm,
           \mathit, \mathtt) which do affect the following group,
           e.g., $\mathcal{R}$.

           Other new features include:

           - Math may be interspersed with non-math text.  Any text
             with an even number of $'s (non-escaped) will be sent to
             the mathtext parser for layout.

           - Sub/superscripts are less likely to accidentally overlap.

           - Support for sub/superscripts in either order, e.g., $x^i_j$
             and $x_j^i$ are equivalent.

           - Double sub/superscripts (e.g., $x_i_j$) are considered
             ambiguous and raise an exception.  Use braces to disambiguate.

           - $\frac{x}{y}$ can be used for displaying fractions.

           - $\sqrt[3]{x}$ can be used to display the radical symbol
             with a root number and body.

           - $\left(\frac{x}{y}\right)$ may be used to create
             parentheses and other delimiters that automatically
             resize to the height of their contents.

           - Spacing around operators etc. is now generally more like
             TeX.

           - Added support (and fonts) for boldface (\bf) and
             sans-serif (\sf) symbols.

           - Log-like function name shortcuts are supported.  For
             example, $\sin(x)$ may be used instead of ${\rm sin}(x)$

           - Limited use of kerning for the easy case (same font)

           Behind the scenes, the pyparsing.py module used for doing
           the math parsing was updated to the latest stable version
           (1.4.6).  A lot of duplicate code was refactored out of the
           Font classes.

           - MGD

2007-07-19 completed numpification of most trivial cases - NN

2007-07-19 converted non-numpy relicts throughout the code - NN

2007-07-19 replaced the Python code in numerix/ by a minimal wrapper around
           numpy that explicitly mentions all symbols that need to be
           addressed for further numpification - NN

2007-07-18 make usetex respect changes to rcParams. texmanager used to
           only configure itself when it was created, now it
           reconfigures when rcParams are changed. Thank you Alexander
           Schmolck for contributing a patch - DSD

2007-07-17 added validation to setting and changing rcParams - DSD

2007-07-17 bugfix segfault in transforms module. Thanks Ben North for
           the patch. - ADS

2007-07-16 clean up some code in ticker.ScalarFormatter, use unicode to
           render multiplication sign in offset ticklabel - DSD

2007-07-16 fixed a formatting bug in ticker.ScalarFormatter's scientific
           notation (10^0 was being rendered as 10 in some cases) - DSD

2007-07-13 Add MPL_isfinite64() and MPL_isinf64() for testing
           doubles in (the now misnamed) MPL_isnan.h. - ADS

2007-07-13 The matplotlib._isnan module removed (use numpy.isnan) - ADS

2007-07-13 Some minor cleanups in _transforms.cpp - ADS

2007-07-13 Removed the rest of the numerix extension code detritus,
           numpified axes.py, and cleaned up the imports in axes.py
           - JDH

2007-07-13 Added legend.loc as configurable option that could in
           future default to 'best'. - NN

2007-07-12 Bugfixes in mlab.py to coerce inputs into numpy arrays. -ADS

2007-07-11 Added linespacing kwarg to text.Text - EF

2007-07-11 Added code to store font paths in SVG files. - MGD

2007-07-10 Store subset of TTF font as a Type 3 font in PDF files. - MGD

2007-07-09 Store subset of TTF font as a Type 3 font in PS files. - MGD

2007-07-09 Applied Paul's pick restructure pick and add pickers,
           sourceforge patch 1749829 - JDH


2007-07-09 Applied Allan's draw_lines agg optimization. JDH


2007-07-08 Applied Carl Worth's patch to fix cairo draw_arc - SC

2007-07-07 fixed bug 1712099: xpdf distiller on windows - DSD

2007-06-30 Applied patches to tkagg, gtk, and wx backends to reduce
           memory leakage.  Patches supplied by Mike Droettboom;
           see tracker numbers 1745400, 1745406, 1745408.
           Also made unit/memleak_gui.py more flexible with
           command-line options. - EF

2007-06-30 Split defaultParams into separate file rcdefaults (together with
           validation code). Some heavy refactoring was necessary to do so,
           but the overall behavior should be the same as before. - NN

2007-06-27 Added MPLCONFIGDIR for the default location for mpl data
           and configuration.  useful for some apache installs where
           HOME is not writable.  Tried to clean up the logic in
           _get_config_dir to support non-writable HOME where are
           writable HOME/.matplotlib already exists - JDH

2007-06-27 Fixed locale bug reported at
           http://sourceforge.net/tracker/index.php?func=detail&aid=1744154&group_id=80706&atid=560720
           by adding a cbook.unicode_safe function - JDH

2007-06-27 Applied Micheal's tk savefig bugfix described at
           http://sourceforge.net/tracker/index.php?func=detail&aid=1716732&group_id=80706&atid=560720
           Thanks Michael!


2007-06-27 Patch for get_py2exe_datafiles() to work with new directory
           layout. (Thanks Tocer and also Werner Bruhin.) -ADS


2007-06-27 Added a scroll event to the mpl event handling system and
           implemented it for backends GTK* -- other backend
           users/developers/maintainers, please add support for your
           backend. - JDH

2007-06-25 Changed default to clip=False in colors.Normalize;
           modified ColorbarBase for easier colormap display - EF

2007-06-13 Added maskedarray option to rc, numerix - EF

2007-06-11 Python 2.5 compatibility fix for mlab.py - EF

2007-06-10 In matplotlibrc file, use 'dashed' | 'solid' instead
           of a pair of floats for contour.negative_linestyle - EF

2007-06-08 Allow plot and fill fmt string to be any mpl string
            colorspec - EF

2007-06-08 Added gnuplot file plotfile function to pylab -- see
           examples/plotfile_demo.py - JDH

2007-06-07 Disable build of numarray and Numeric extensions for
           internal MPL use and the numerix layer. - ADS

2007-06-07 Added csv2rec to matplotlib.mlab to support automatically
           converting csv files to record arrays using type
           introspection, and turned on native datetime support using
           the new units support in matplotlib.dates.  See
           examples/loadrec.py !  JDH

2007-06-07 Simplified internal code of _auto_legend_data - NN

2007-06-04 Added labeldistance arg to Axes.pie to control the raidal
           distance of the wedge labels - JDH

2007-06-03 Turned mathtext in SVG into single <text> with multiple <tspan>
           objects (easier to edit in inkscape). - NN

----------------------------

2007-06-02 Released 0.90.1 at revision 3352

2007-06-02 Display only meaningful labels when calling legend()
           without args. - NN

2007-06-02 Have errorbar follow the color cycle even if line is not plotted.
           Suppress plotting of errorbar caps for capsize=0. - NN

2007-06-02 Set markers to same alpha value as line. - NN

2007-06-02 Fix mathtext position in svg backend. - NN

2007-06-01 Deprecate Numeric and numarray for use as numerix. Props to
           Travis -- job well done. - ADS

2007-05-18 Added LaTeX unicode support. Enable with the
           'text.latex.unicode' rcParam. This requires the ucs and
           inputenc LaTeX packages. - ADS

2007-04-23 Fixed some problems with polar -- added general polygon
           clipping to clip the lines a nd grids to the polar axes.
           Added support for set_rmax to easily change the maximum
           radial grid.  Added support for polar legend - JDH

2007-04-16 Added Figure.autofmt_xdate to handle adjusting the bottom
           and rotating the tick labels for date plots when the ticks
           often overlap - JDH

2007-04-09 Beginnings of usetex support for pdf backend. -JKS

2007-04-07 Fixed legend/LineCollection bug. Added label support
           to collections. - EF

2007-04-06 Removed deprecated support for a float value as a gray-scale;
           now it must be a string, like '0.5'.  Added alpha kwarg to
           ColorConverter.to_rgba_list. - EF

2007-04-06 Fixed rotation of ellipses in pdf backend
           (sf bug #1690559) -JKS

2007-04-04 More matshow tweaks; documentation updates; new method
           set_bounds() for formatters and locators. - EF

2007-04-02 Fixed problem with imshow and matshow of integer arrays;
           fixed problems with changes to color autoscaling. - EF

2007-04-01 Made image color autoscaling work correctly with
           a tracking colorbar; norm.autoscale now scales
           unconditionally, while norm.autoscale_None changes
           only None-valued vmin, vmax. - EF

2007-03-31 Added a qt-based subplot-adjustment dialog - DSD

2007-03-30 Fixed a bug in backend_qt4, reported on mpl-dev - DSD

2007-03-26 Removed colorbar_classic from figure.py; fixed bug in
           Figure.clf() in which _axobservers was not getting
           cleared.  Modernization and cleanups. - EF

2007-03-26 Refactored some of the units support -- units now live in
           the respective x and y Axis instances.  See also
           API_CHANGES for some alterations to the conversion
           interface.  JDH

2007-03-25 Fix masked array handling in quiver.py for numpy. (Numeric
           and numarray support for masked arrays is broken in other
           ways when using quiver. I didn't pursue that.) - ADS

2007-03-23 Made font_manager.py close opened files. - JKS

2007-03-22 Made imshow default extent match matshow - EF

2007-03-22 Some more niceties for xcorr -- a maxlags option, normed
           now works for xcorr as well as axorr, usevlines is
           supported, and a zero correlation hline is added.  See
           examples/xcorr_demo.py.  Thanks Sameer for the patch.  -
           JDH

2007-03-21 Axes.vlines and Axes.hlines now create and returns a
           LineCollection, not a list of lines.  This is much faster.
           The kwarg signature has changed, so consult the docs.
           Modified Axes.errorbar which uses vlines and hlines.  See
           API_CHANGES; the return signature for these three functions
           is now different

2007-03-20 Refactored units support and added new examples - JDH

2007-03-19 Added Mike's units patch - JDH

2007-03-18 Matshow as an Axes method; test version matshow1() in
           pylab; added 'integer' Boolean kwarg to MaxNLocator
           initializer to force ticks at integer locations. - EF

2007-03-17 Preliminary support for clipping to paths agg - JDH

2007-03-17 Text.set_text() accepts anything convertible with '%s' - EF

2007-03-14 Add masked-array support to hist. - EF

2007-03-03 Change barh to take a kwargs dict and pass it to bar.
           Fixes sf bug #1669506.

2007-03-02 Add rc parameter pdf.inheritcolor, which disables all
           color-setting operations in the pdf backend. The idea is
           that you include the resulting file in another program and
           set the colors (both stroke and fill color) there, so you
           can use the same pdf file for e.g., a paper and a
           presentation and have them in the surrounding color. You
           will probably not want to draw figure and axis frames in
           that case, since they would be filled in the same color. - JKS

2007-02-26 Prevent building _wxagg.so with broken Mac OS X wxPython. - ADS

2007-02-23 Require setuptools for Python 2.3 - ADS

2007-02-22 WXAgg accelerator updates - KM
           WXAgg's C++ accelerator has been fixed to use the correct wxBitmap
           constructor.

           The backend has been updated to use new wxPython functionality to
           provide fast blit() animation without the C++ accelerator.  This
           requires wxPython 2.8 or later.  Previous versions of wxPython can
           use the C++ acclerator or the old pure Python routines.

           setup.py no longer builds the C++ accelerator when wxPython >= 2.8
           is present.

           The blit() method is now faster regardless of which agg/wxPython
           conversion routines are used.

2007-02-21 Applied the PDF backend patch by Nicolas Grilly.
           This impacts several files and directories in matplotlib:

           - Created the directory lib/matplotlib/mpl-data/fonts/pdfcorefonts,
             holding AFM files for the 14 PDF core fonts. These fonts are
             embedded in every PDF viewing application.

           - setup.py: Added the directory pdfcorefonts to package_data.

           - lib/matplotlib/__init__.py: Added the default parameter
             'pdf.use14corefonts'. When True, the PDF backend uses
             only the 14 PDF core fonts.

           - lib/matplotlib/afm.py: Added some keywords found in
             recent AFM files. Added a little workaround to handle
             Euro symbol.

           - lib/matplotlib/fontmanager.py: Added support for the 14
             PDF core fonts. These fonts have a dedicated cache (file
             pdfcorefont.cache), not the same as for other AFM files
             (file .afmfont.cache). Also cleaned comments to conform
             to CODING_GUIDE.

           - lib/matplotlib/backends/backend_pdf.py:
             Added support for 14 PDF core fonts.
             Fixed some issues with incorrect character widths and
             encodings (works only for the most common encoding,
             WinAnsiEncoding, defined by the official PDF Reference).
             Removed parameter 'dpi' because it causes alignment issues.

           -JKS (patch by Nicolas Grilly)

2007-02-17 Changed ft2font.get_charmap, and updated all the files where
           get_charmap is mentioned - ES

2007-02-13 Added barcode demo- JDH

2007-02-13 Added binary colormap to cm - JDH

2007-02-13 Added twiny to pylab - JDH

2007-02-12 Moved data files into lib/matplotlib so that setuptools'
           develop mode works. Re-organized the mpl-data layout so
           that this source structure is maintained in the
           installation. (i.e., the 'fonts' and 'images'
           sub-directories are maintained in site-packages.)  Suggest
           removing site-packages/matplotlib/mpl-data and
           ~/.matplotlib/ttffont.cache before installing - ADS

2007-02-07 Committed Rob Hetland's patch for qt4: remove
           references to text()/latin1(), plus some improvements
           to the toolbar layout - DSD

---------------------------

2007-02-06 Released 0.90.0 at revision 3003

2007-01-22 Extended the new picker API to text, patches and patch
           collections.  Added support for user customizable pick hit
           testing and attribute tagging of the PickEvent - Details
           and examples in examples/pick_event_demo.py - JDH

2007-01-16 Begun work on a new pick API using the mpl event handling
           frameowrk.  Artists will define their own pick method with
           a configurable epsilon tolerance and return pick attrs.
           All artists that meet the tolerance threshold will fire a
           PickEvent with artist dependent attrs; e.g., a Line2D can set
           the indices attribute that shows the indices into the line
           that are within epsilon of the pick point.  See
           examples/pick_event_demo.py.  The implementation of pick
           for the remaining Artists remains to be done, but the core
           infrastructure at the level of event handling is in place
           with a proof-of-concept implementation for Line2D - JDH

2007-01-16 src/_image.cpp: update to use Py_ssize_t (for 64-bit systems).
           Use return value of fread() to prevent warning messages - SC.

2007-01-15 src/_image.cpp: combine buffer_argb32() and buffer_bgra32() into
           a new method color_conv(format) - SC

2007-01-14 backend_cairo.py: update draw_arc() so that
           examples/arctest.py looks correct - SC

2007-01-12 backend_cairo.py: enable clipping. Update draw_image() so that
           examples/contour_demo.py looks correct - SC

2007-01-12 backend_cairo.py: fix draw_image() so that examples/image_demo.py
           now looks correct - SC

2007-01-11 Added Axes.xcorr and Axes.acorr to plot the cross
           correlation of x vs y or the autocorrelation of x.  pylab
           wrappers also provided.  See examples/xcorr_demo.py - JDH

2007-01-10 Added "Subplot.label_outer" method.  It will set the
           visibility of the ticklabels so that yticklabels are only
           visible in the first column and xticklabels are only
           visible in the last row - JDH

2007-01-02 Added additional kwarg documentation - JDH

2006-12-28 Improved error message for nonpositive input to log
           transform; added log kwarg to bar, barh, and hist,
           and modified bar method to behave sensibly by default
           when the ordinate has a log scale.  (This only works
           if the log scale is set before or by the call to bar,
           hence the utility of the log kwarg.) - EF

2006-12-27 backend_cairo.py: update draw_image() and _draw_mathtext() to work
           with numpy - SC

2006-12-20 Fixed xpdf dependency check, which was failing on windows.
           Removed ps2eps dependency check. - DSD

2006-12-19 Added Tim Leslie's spectral patch - JDH

2006-12-17 Added rc param 'axes.formatter.limits' to control
           the default threshold for switching to scientific
           notation. Added convenience method
           Axes.ticklabel_format() for turning scientific notation
           on or off on either or both axes. - EF

2006-12-16 Added ability to turn control scientific notation
           in ScalarFormatter - EF

2006-12-16 Enhanced boxplot to handle more flexible inputs - EF

2006-12-13 Replaced calls to where() in colors.py with much faster
           clip() and putmask() calls; removed inappropriate
           uses of getmaskorNone (which should be needed only
           very rarely); all in response to profiling by
           David Cournapeau.  Also fixed bugs in my 2-D
           array support from 12-09. - EF

2006-12-09 Replaced spy and spy2 with the new spy that combines
           marker and image capabilities - EF

2006-12-09 Added support for plotting 2-D arrays with plot:
           columns are plotted as in Matlab - EF

2006-12-09 Added linewidth kwarg to bar and barh; fixed arg
           checking bugs - EF

2006-12-07 Made pcolormesh argument handling match pcolor;
           fixed kwarg handling problem noted by Pierre GM - EF

2006-12-06 Made pcolor support vector X and/or Y instead of
           requiring 2-D arrays - EF

2006-12-05 Made the default Artist._transform None (rather than
           invoking identity_transform for each artist only to have it
           overridden later).  Use artist.get_transform() rather than
           artist._transform, even in derived classes, so that the
           default transform will be created lazily as needed - JDH

2006-12-03 Added LogNorm to colors.py as illustrated by
           examples/pcolor_log.py, based on suggestion by
           Jim McDonald.  Colorbar modified to handle LogNorm.
           Norms have additional "inverse" method. - EF

2006-12-02 Changed class names in colors.py to match convention:
           normalize -> Normalize, no_norm -> NoNorm.  Old names
           are still available.
           Changed __init__.py rc defaults to match those in
           matplotlibrc - EF

2006-11-22 Fixed bug in set_*lim that I had introduced on 11-15 - EF

2006-11-22 Added examples/clippedline.py, which shows how to clip line
           data based on view limits -- it also changes the marker
           style when zoomed in - JDH

2006-11-21 Some spy bug-fixes and added precision arg per Robert C's
           suggestion  - JDH

2006-11-19 Added semi-automatic docstring generation detailing all the
           kwargs that functions take using the artist introspection
           tools; e.g., 'help text now details the scatter kwargs
           that control the Text properties  - JDH

2006-11-17 Removed obsolete scatter_classic, leaving a stub to
           raise NotImplementedError; same for pcolor_classic - EF

2006-11-15 Removed obsolete pcolor_classic - EF

2006-11-15 Fixed 1588908 reported by Russel Owen; factored
           nonsingular method out of ticker.py, put it into
           transforms.py as a function, and used it in
           set_xlim and set_ylim. - EF

2006-11-14 Applied patch 1591716 by Ulf Larssen to fix a bug in
           apply_aspect.  Modified and applied patch
           1594894 by mdehoon to fix bugs and improve
           formatting in lines.py. Applied patch 1573008
           by Greg Willden to make psd etc. plot full frequency
           range for complex inputs. - EF

2006-11-14 Improved the ability of the colorbar to track
           changes in corresponding image, pcolor, or
           contourf. - EF

2006-11-11 Fixed bug that broke Numeric compatibility;
           added support for alpha to colorbar.  The
           alpha information is taken from the mappable
           object, not specified as a kwarg. - EF

2006-11-05 Added broken_barh function for makring a sequence of
           horizontal bars broken by gaps -- see examples/broken_barh.py

2006-11-05 Removed lineprops and markerprops from the Annotation code
           and replaced them with an arrow configurable with kwarg
           arrowprops.  See examples/annotation_demo.py - JDH

2006-11-02 Fixed a pylab subplot bug that was causing axes to be
           deleted with hspace or wspace equals zero in
           subplots_adjust - JDH

2006-10-31 Applied axes3d patch 1587359
           http://sourceforge.net/tracker/index.php?func=detail&aid=1587359&group_id=80706&atid=560722
           JDH

-------------------------

2006-10-26 Released 0.87.7 at revision 2835

2006-10-25 Made "tiny" kwarg in Locator.nonsingular much smaller - EF

2006-10-17 Closed sf bug 1562496 update line props dash/solid/cap/join
           styles - JDH

2006-10-17 Complete overhaul of the annotations API and example code -
           See matplotlib.text.Annotation and
           examples/annotation_demo.py JDH

2006-10-12 Committed Manuel Metz's StarPolygon code and
           examples/scatter_star_poly.py - JDH


2006-10-11 commented out all default values in matplotlibrc.template
           Default values should generally be taken from defaultParam in
           __init__.py - the file matplotlib should only contain those values
           that the user wants to explicitly change from the default.
           (see thread "marker color handling" on matplotlib-devel)

2006-10-10 Changed default comment character for load to '#' - JDH

2006-10-10 deactivated rcfile-configurability of markerfacecolor
           and markeredgecolor. Both are now hardcoded to the special value
           'auto' to follow the line color. Configurability at run-time
           (using function arguments) remains functional. - NN

2006-10-07 introduced dummy argument magnification=1.0 to
           FigImage.make_image to satisfy unit test figimage_demo.py
           The argument is not yet handled correctly, which should only
           show up when using non-standard DPI settings in PS backend,
           introduced by patch #1562394. - NN

2006-10-06 add backend-agnostic example: simple3d.py - NN

2006-09-29 fix line-breaking for SVG-inline images (purely cosmetic) - NN

2006-09-29 reworked set_linestyle and set_marker
           markeredgecolor and markerfacecolor now default to
           a special value "auto" that keeps the color in sync with
           the line color
           further, the intelligence of axes.plot is cleaned up,
           improved and simplified. Complete compatibility cannot be
           guaranteed, but the new behavior should be much more predictable
           (see patch #1104615 for details) - NN

2006-09-29 changed implementation of clip-path in SVG to work around a
           limitation in inkscape - NN

2006-09-29 added two options to matplotlibrc:
           svg.image_inline
           svg.image_noscale
           see patch #1533010 for details - NN

2006-09-29 axes.py: cleaned up kwargs checking - NN

2006-09-29 setup.py: cleaned up setup logic - NN

2006-09-29 setup.py: check for required pygtk versions, fixes bug #1460783 - SC

---------------------------------

2006-09-27 Released 0.87.6 at revision 2783

2006-09-24 Added line pointers to the Annotation code, and a pylab
           interface.  See matplotlib.text.Annotation,
           examples/annotation_demo.py and
           examples/annotation_demo_pylab.py - JDH

2006-09-18 mathtext2.py: The SVG backend now supports the same things that
           the AGG backend does. Fixed some bugs with rendering, and out of
           bounds errors in the AGG backend - ES. Changed the return values
           of math_parse_s_ft2font_svg to support lines (fractions etc.)

2006-09-17 Added an Annotation class to facilitate annotating objects
           and an examples file examples/annotation_demo.py.  I want
           to add dash support as in TextWithDash, but haven't decided
           yet whether inheriting from TextWithDash is the right base
           class or if another approach is needed - JDH

------------------------------

2006-09-05 Released 0.87.5 at revision 2761

2006-09-04 Added nxutils for some numeric add-on extension code --
                   specifically a better/more efficient inside polygon tester (see
                   unit/inside_poly_*.py) - JDH

2006-09-04 Made bitstream fonts the rc default - JDH

2006-08-31 Fixed alpha-handling bug in ColorConverter, affecting
           collections in general and contour/contourf in
           particular. - EF

2006-08-30 ft2font.cpp: Added draw_rect_filled method (now used by mathtext2
           to draw the fraction bar) to FT2Font - ES

2006-08-29 setupext.py: wrap calls to tk.getvar() with str(). On some
           systems, getvar returns a Tcl_Obj instead of a string - DSD

2006-08-28 mathtext2.py: Sub/superscripts can now be complex (i.e.
           fractions etc.). The demo is also updated - ES

2006-08-28 font_manager.py: Added /usr/local/share/fonts to list of
           X11 font directories - DSD

2006-08-28 mahtext2.py: Initial support for complex fractions. Also,
           rendering is now completely separated from parsing. The
           sub/superscripts now work better.
           Updated the mathtext2_demo.py - ES

2006-08-27 qt backends: don't create a QApplication when backend is
           imported, do it when the FigureCanvasQt is created. Simplifies
           applications where mpl is embedded in qt. Updated
           embedding_in_qt* examples - DSD

2006-08-27 mahtext2.py: Now the fonts are searched in the OS font dir and
           in the mpl-data dir. Also env is not a dict anymore. - ES

2006-08-26 minor changes to __init__.py, mathtex2_demo.py. Added matplotlibrc
           key "mathtext.mathtext2" (removed the key "mathtext2") - ES

2006-08-21 mathtext2.py: Initial support for fractions
           Updated the mathtext2_demo.py
           _mathtext_data.py: removed "\" from the unicode dicts
           mathtext.py: Minor modification (because of _mathtext_data.py)- ES

2006-08-20 Added mathtext2.py: Replacement for mathtext.py. Supports _ ^,
           \rm, \cal etc., \sin, \cos etc., unicode, recursive nestings,
           inline math mode. The only backend currently supported is Agg
           __init__.py: added new rc params for mathtext2
           added mathtext2_demo.py example - ES

2006-08-19 Added embedding_in_qt4.py example - DSD

2006-08-11 Added scale free Ellipse patch for Agg - CM

2006-08-10 Added converters to and from julian dates to matplotlib.dates
           (num2julian and julian2num) - JDH

2006-08-08 Fixed widget locking so multiple widgets could share the
           event handling - JDH

2006-08-07 Added scale free Ellipse patch to SVG and PS - CM

2006-08-05 Re-organized imports in numerix for numpy 1.0b2 -- TEO

2006-08-04 Added draw_markers to PDF backend. - JKS

2006-08-01 Fixed a bug in postscript's rendering of dashed lines - DSD

2006-08-01 figure.py: savefig() update docstring to add support for 'format'
           argument.
           backend_cairo.py: print_figure() add support 'format' argument. - SC

2006-07-31 Don't let postscript's xpdf distiller compress images - DSD

2006-07-31 Added shallowcopy() methods to all Transformations;
           removed copy_bbox_transform and copy_bbox_transform_shallow
           from transforms.py;
           added offset_copy() function to transforms.py to
           facilitate positioning artists with offsets.
           See examples/transoffset.py. - EF

2006-07-31 Don't let postscript's xpdf distiller compress images - DSD

2006-07-29 Fixed numerix polygon bug reported by Nick Fotopoulos.
           Added inverse_numerix_xy() transform method.
           Made autoscale_view() preserve axis direction
           (e.g., increasing down).- EF

2006-07-28 Added shallow bbox copy routine for transforms -- mainly
           useful for copying transforms to apply offset to. - JDH

2006-07-28 Added resize method to FigureManager class
           for Qt and Gtk backend - CM

2006-07-28 Added subplots_adjust button to Qt backend - CM

2006-07-26 Use numerix more in collections.
           Quiver now handles masked arrays. - EF

2006-07-22 Fixed bug #1209354 - DSD

2006-07-22 make scatter() work with the kwarg "color". Closes bug
           1285750 - DSD

2006-07-20 backend_cairo.py: require pycairo 1.2.0.
           print_figure() update to output SVG using cairo.

2006-07-19 Added blitting for Qt4Agg - CM

2006-07-19 Added lasso widget and example examples/lasso_demo.py - JDH

2006-07-18 Added blitting for QtAgg backend - CM

2006-07-17 Fixed bug #1523585: skip nans in semilog plots - DSD

2006-07-12 Add support to render the scientific notation label
           over the right-side y-axis - DSD

------------------------------

2006-07-11 Released 0.87.4 at revision 2558

2006-07-07 Fixed a usetex bug with older versions of latex - DSD

2006-07-07 Add compatibility for NumPy 1.0 - TEO

2006-06-29 Added a Qt4Agg backend. Thank you James Amundson - DSD

2006-06-26 Fixed a usetex bug. On windows, usetex will prcess
           postscript output in the current directory rather than
           in a temp directory. This is due to the use of spaces
           and tildes in windows paths, which cause problems with
           latex. The subprocess module is no longer used. - DSD

2006-06-22 Various changes to bar(), barh(), and hist().
           Added 'edgecolor' keyword arg to bar() and barh().
           The x and y args in barh() have been renamed to width
           and bottom respectively, and their order has been swapped
           to maintain a (position, value) order ala matlab. left,
           height, width and bottom args can now all be scalars or
           sequences. barh() now defaults to edge alignment instead
           of center alignment. Added a keyword arg 'align' to bar(),
           barh() and hist() that controls between edge or center bar
           alignment. Fixed ignoring the rcParams['patch.facecolor']
           for bar color in bar() and barh(). Fixed ignoring the
           rcParams['lines.color'] for error bar color in bar()
           and barh(). Fixed a bug where patches would be cleared
           when error bars were plotted if rcParams['axes.hold']
           was False. - MAS

2006-06-22 Added support for numerix 2-D arrays as alternatives to
           a sequence of (x,y) tuples for specifying paths in
           collections, quiver, contour, pcolor, transforms.
           Fixed contour bug involving setting limits for
           color mapping.  Added numpy-style all() to numerix. - EF

2006-06-20 Added custom FigureClass hook to pylab interface - see
           examples/custom_figure_class.py

2006-06-16 Added colormaps from gist (gist_earth, gist_stern,
           gist_rainbow, gist_gray, gist_yarg, gist_heat, gist_ncar) - JW

2006-06-16 Added a pointer to parent in figure canvas so you can
           access the container with fig.canvas.manager.  Useful if
           you want to set the window title, e.g., in gtk
           fig.canvas.manager.window.set_title, though a GUI neutral
           method would be preferable JDH

2006-06-16 Fixed colorbar.py to handle indexed colors (i.e.,
           norm = no_norm()) by centering each colored region
           on its index. - EF

2006-06-15 Added scalex and scaley to Axes.autoscale_view to support
           selective autoscaling just the x or y axis, and supported
           these command in plot so you can say plot(something,
           scaley=False) and just the x axis will be autoscaled.
           Modified axvline and axhline to support this, so for
           example axvline will no longer autoscale the y axis. JDH

2006-06-13 Fix so numpy updates are backward compatible - TEO

2006-06-12 Updated numerix to handle numpy restructuring of
           oldnumeric - TEO

2006-06-12 Updated numerix.fft to handle numpy restructuring
           Added ImportError to numerix.linear_algebra for numpy -TEO

2006-06-11 Added quiverkey command to pylab and Axes, using
           QuiverKey class in quiver.py.  Changed pylab and Axes
           to use quiver2 if possible, but drop back to the
           newly-renamed quiver_classic if necessary.  Modified
           examples/quiver_demo.py to illustrate the new quiver
           and quiverkey.  Changed LineCollection implementation
           slightly to improve compatibility with PolyCollection. - EF

2006-06-11 Fixed a usetex bug for windows, running latex on files
           with spaces in their names or paths was failing - DSD

2006-06-09 Made additions to numerix, changes to quiver to make it
           work with all numeric flavors. - EF

2006-06-09 Added quiver2 function to pylab and method to axes,
           with implementation via a Quiver class in quiver.py.
           quiver2 will replace quiver before the next release;
           it is placed alongside it initially to facilitate
           testing and transition. See also
           examples/quiver2_demo.py. - EF

2006-06-08 Minor bug fix to make ticker.py draw proper minus signs
           with usetex - DSD

-----------------------

2006-06-06 Released 0.87.3 at revision 2432

2006-05-30 More partial support for polygons with outline or fill,
           but not both.  Made LineCollection inherit from
           ScalarMappable. - EF

2006-05-29 Yet another revision of aspect-ratio handling. - EF

2006-05-27 Committed a patch to prevent stroking zero-width lines in
           the svg backend - DSD

2006-05-24 Fixed colorbar positioning bug identified by Helge
           Avlesen, and improved the algorithm; added a 'pad'
           kwarg to control the spacing between colorbar and
           parent axes. - EF

2006-05-23 Changed color handling so that collection initializers
           can take any mpl color arg or sequence of args; deprecated
           float as grayscale, replaced by string representation of
           float. - EF

2006-05-19 Fixed bug: plot failed if all points were masked - EF

2006-05-19 Added custom symbol option to scatter - JDH

2006-05-18 New example, multi_image.py; colorbar fixed to show
           offset text when the ScalarFormatter is used; FixedFormatter
           augmented to accept and display offset text. - EF

2006-05-14 New colorbar; old one is renamed to colorbar_classic.
           New colorbar code is in colorbar.py, with wrappers in
           figure.py and pylab.py.
           Fixed aspect-handling bug reported by Michael Mossey.
           Made backend_bases.draw_quad_mesh() run.- EF

2006-05-08 Changed handling of end ranges in contourf: replaced
           "clip-ends" kwarg with "extend".  See docstring for
           details. -EF

2006-05-08 Added axisbelow to rc - JDH

2006-05-08 If using PyGTK require version 2.2+ - SC

2006-04-19 Added compression support to PDF backend, controlled by
           new pdf.compression rc setting. - JKS

2006-04-19 Added Jouni's PDF backend

2006-04-18 Fixed a bug that caused agg to not render long lines

2006-04-16 Masked array support for pcolormesh; made pcolormesh support the
           same combinations of X,Y,C dimensions as pcolor does;
           improved (I hope) description of grid used in pcolor,
           pcolormesh. - EF

2006-04-14 Reorganized axes.py - EF

2006-04-13 Fixed a bug Ryan found using usetex with sans-serif fonts and
           exponential tick labels - DSD

2006-04-11 Refactored backend_ps and backend_agg to prevent module-level
           texmanager imports. Now these imports only occur if text.usetex
           rc setting is true - DSD

2006-04-10 Committed changes required for building mpl on win32
           platforms with visual studio.  This allows wxpython
           blitting for fast animations. - CM

2006-04-10 Fixed an off-by-one bug in Axes.change_geometry.

2006-04-10 Fixed bug in pie charts where wedge wouldn't have label in
           legend. Submitted by Simon Hildebrandt. - ADS

2006-05-06 Usetex makes temporary latex and dvi files in a temporary
           directory, rather than in the user's current working
           directory - DSD

2006-04-05 Apllied Ken's wx deprecation warning patch closing sf patch
           #1465371 - JDH

2006-04-05 Added support for the new API in the postscript backend.
           Allows values to be masked using nan's, and faster file
           creation - DSD

2006-04-05 Use python's subprocess module for usetex calls to
           external programs. subprocess catches when they exit
           abnormally so an error can be raised. - DSD

2006-04-03 Fixed the bug in which widgets would not respond to
           events.  This regressed the twinx functionality, so I
           also updated subplots_adjust to update axes that share
           an x or y with a subplot instance. - CM

2006-04-02 Moved PBox class to transforms and deleted pbox.py;
           made pylab axis command a thin wrapper for Axes.axis;
           more tweaks to aspect-ratio handling; fixed Axes.specgram
           to account for the new imshow default of unit aspect
           ratio; made contour set the Axes.dataLim. - EF

2006-03-31 Fixed the Qt "Underlying C/C++ object deleted" bug. - JRE

2006-03-31 Applied Vasily Sulatskov's Qt Navigation Toolbar enhancement. - JRE

2006-03-31 Ported Norbert's rewriting of Halldor's stineman_interp
           algorithm to make it numerix compatible and added code to
           matplotlib.mlab.  See examples/interp_demo.py - JDH

2006-03-30 Fixed a bug in aspect ratio handling; blocked potential
           crashes when panning with button 3; added axis('image')
           support. - EF

2006-03-28 More changes to aspect ratio handling; new PBox class
           in new file pbox.py to facilitate resizing and repositioning
           axes; made PolarAxes maintain unit aspect ratio. - EF

2006-03-23 Refactored TextWithDash class to inherit from, rather than
           delegate to, the Text class. Improves object inspection
           and closes bug # 1357969 - DSD

2006-03-22 Improved aspect ratio handling, including pylab interface.
           Interactive resizing, pan, zoom of images and plots
           (including panels with a shared axis) should work.
           Additions and possible refactoring are still likely. - EF

2006-03-21 Added another colorbrewer colormap (RdYlBu) - JSWHIT

2006-03-21 Fixed tickmarks for logscale plots over very large ranges.
           Closes bug # 1232920 - DSD

2006-03-21 Added Rob Knight's arrow code; see examples/arrow_demo.py - JDH

2006-03-20 Added support for masking values with nan's, using ADS's
           isnan module and the new API. Works for \*Agg backends - DSD

2006-03-20 Added contour.negative_linestyle rcParam - ADS

2006-03-20 Added _isnan extension module to test for nan with Numeric
           - ADS

2006-03-17 Added Paul and Alex's support for faceting with quadmesh
           in sf patch 1411223 - JDH

2006-03-17 Added Charle Twardy's pie patch to support colors=None.
           Closes sf patch 1387861 - JDH

2006-03-17 Applied sophana's patch to support overlapping axes with
           toolbar navigation by toggling activation with the 'a' key.
           Closes sf patch 1432252 - JDH

2006-03-17 Applied Aarre's linestyle patch for backend EMF; closes sf
           patch 1449279 - JDH

2006-03-17 Applied  Jordan Dawe's patch to support kwarg properties
           for grid lines in the grid command.  Closes sf patch
           1451661 - JDH

2006-03-17 Center postscript output on page when using usetex - DSD

2006-03-17 subprocess module built if Python <2.4 even if subprocess
           can be imported from an egg - ADS

2006-03-17 Added _subprocess.c from Python upstream and hopefully
           enabled building (without breaking) on Windows, although
           not tested. - ADS

2006-03-17 Updated subprocess.py to latest Python upstream and
           reverted name back to subprocess.py - ADS

2006-03-16 Added John Porter's 3D handling code


------------------------

2006-03-16 Released 0.87.2 at revision 2150

2006-03-15 Fixed bug in MaxNLocator revealed by daigos@infinito.it.
           The main change is that Locator.nonsingular now adjusts
           vmin and vmax if they are nearly the same, not just if
           they are equal.  A new kwarg, "tiny", sets the threshold. -
           EF

2006-03-14 Added import of compatibility library for newer numpy
           linear_algebra - TEO

2006-03-12 Extended "load" function to support individual columns and
           moved "load" and "save" into matplotlib.mlab so they can be
           used outside of pylab -- see examples/load_converter.py -
           JDH

2006-03-12 Added AutoDateFormatter and AutoDateLocator submitted
           by James Evans. Try the load_converter.py example for a
           demo. - ADS

2006-03-11 Added subprocess module from python-2.4 - DSD

2006-03-11 Fixed landscape orientation support with the usetex
           option. The backend_ps print_figure method was
           getting complicated, I added a _print_figure_tex
           method to maintain some degree of sanity - DSD

2006-03-11 Added "papertype" savefig kwarg for setting
           postscript papersizes. papertype and ps.papersize
           rc setting can also be set to "auto" to autoscale
           pagesizes - DSD

2006-03-09 Apply P-J's patch to make pstoeps work on windows
           patch report # 1445612 - DSD

2006-03-09 Make backend rc parameter case-insensitive - DSD

2006-03-07 Fixed bug in backend_ps related to C0-C6 papersizes,
           which were causing problems with postscript viewers.
           Supported page sizes include letter, legal, ledger,
           A0-A10, and B0-B10 - DSD

------------------------------------

2006-03-07 Released 0.87.1

2006-03-04 backend_cairo.py:
           fix get_rgb() bug reported by Keith Briggs.
           Require pycairo 1.0.2.
           Support saving png to file-like objects. - SC

2006-03-03 Fixed pcolor handling of vmin, vmax - EF

2006-03-02 improve page sizing with usetex with the latex
           geometry package. Closes bug # 1441629 - DSD

2006-03-02 Fixed dpi problem with usetex png output. Accepted a
           modified version of patch # 1441809 - DSD

2006-03-01 Fixed axis('scaled') to deal with case xmax < xmin - JSWHIT

2006-03-01 Added reversed colormaps (with '_r' appended to name) - JSWHIT

2006-02-27 Improved eps bounding boxes with usetex - DSD

2006-02-27 Test svn commit, again!

2006-02-27 Fixed two dependency checking bugs related to usetex
           on Windows - DSD

2006-02-27 Made the rc deprecation warnings a little more human
           readable.

2006-02-26 Update the previous gtk.main_quit() bug fix to use gtk.main_level()
           - SC

2006-02-24 Implemented alpha support in contour and contourf - EF

2006-02-22 Fixed gtk main quit bug when quit was called before
           mainloop. - JDH

2006-02-22 Small change to colors.py to workaround apparent
           bug in numpy masked array module - JSWHIT

2006-02-22 Fixed bug in ScalarMappable.to_rgba() reported by
           Ray Jones, and fixed incorrect fix found by Jeff
           Whitaker - EF

--------------------------------

2006-02-22 Released 0.87

2006-02-21 Fixed portrait/landscape orientation in postscript backend - DSD

2006-02-21 Fix bug introduced in yesterday's bug fix - SC

2006-02-20 backend_gtk.py FigureCanvasGTK.draw(): fix bug reported by
           David Tremouilles - SC

2006-02-20 Remove the "pygtk.require('2.4')" error from
           examples/embedding_in_gtk2.py - SC

2006-02-18 backend_gtk.py FigureCanvasGTK.draw(): simplify to use (rather than
           duplicate) the expose_event() drawing code - SC

2006-02-12 Added stagger or waterfall plot capability to LineCollection;
           illustrated in examples/collections.py. - EF

2006-02-11 Massive cleanup of the usetex code in the postscript backend. Possibly
           fixed the clipping issue users were reporting with older versions of
           ghostscript - DSD

2006-02-11 Added autolim kwarg to axes.add_collection.  Changed
           collection get_verts() methods accordingly. - EF

2006-02-09 added a temporary rc parameter text.dvipnghack, to allow Mac users to get nice
           results with the usetex option. - DSD

2006-02-09 Fixed a bug related to setting font sizes with the usetex option. - DSD

2006-02-09 Fixed a bug related to usetex's latex code. - DSD

2006-02-09 Modified behavior of font.size rc setting. You should define font.size in pts,
           which will set the "medium" or default fontsize. Special text sizes like axis
           labels or tick labels can be given relative font sizes like small, large,
           x-large, etc. and will scale accordingly. - DSD

2006-02-08 Added py2exe specific datapath check again.  Also added new
           py2exe helper function get_py2exe_datafiles for use in py2exe
           setup.py scripts. - CM

2006-02-02 Added box function to pylab

2006-02-02 Fixed a problem in setupext.py, tk library formatted in unicode
           caused build problems - DSD

2006-02-01 Dropped TeX engine support in usetex to focus on LaTeX. - DSD

2006-01-29 Improved usetex option to respect the serif, sans-serif, monospace,
           and cursive rc settings. Removed the font.latex.package rc setting,
           it is no longer required - DSD

2006-01-29 Fixed tex's caching to include font.family rc information - DSD

2006-01-29 Fixed subpixel rendering bug in \*Agg that was causing
           uneven gridlines - JDH

2006-01-28 Added fontcmd to backend_ps's RendererPS.draw_tex, to support other
           font families in eps output - DSD

2006-01-28 Added MaxNLocator to ticker.py, and changed contour.py to
           use it by default. - EF

2006-01-28 Added fontcmd to backend_ps's RendererPS.draw_tex, to support other
           font families in eps output - DSD

2006-01-27 Buffered reading of matplotlibrc parameters in order to allow
           'verbose' settings to be processed first (allows verbose.report
           during rc validation process) - DSD

2006-01-27 Removed setuptools support from setup.py and created a
           separate setupegg.py file to replace it. - CM

2006-01-26 Replaced the ugly datapath logic with a cleaner approach from
           http://wiki.python.org/moin/DistutilsInstallDataScattered.
           Overrides the install_data command. - CM

2006-01-24 Don't use character typecodes in cntr.c --- changed to use
           defined typenumbers instead. - TEO

2006-01-24 Fixed some bugs in usetex's and ps.usedistiller's dependency

2006-01-24 Added masked array support to scatter - EF

2006-01-24 Fixed some bugs in usetex's and ps.usedistiller's dependency
           checking - DSD

-------------------------------

2006-01-24 Released 0.86.2

2006-01-20 Added a converters dict to pylab load to convert selected
           coloumns to float -- especially useful for files with date
           strings, uses a datestr2num converter - JDH

2006-01-20 Added datestr2num to matplotlib dates to convert a string
           or sequence of strings to a matplotlib datenum

2006-01-18 Added quadrilateral pcolormesh patch 1409190 by Alex Mont
           and Paul Kienzle -- this is \*Agg only for now.  See
           examples/quadmesh_demo.py - JDH

2006-01-18 Added Jouni's boxplot patch - JDH

2006-01-18 Added comma delimiter for pylab save - JDH

2006-01-12 Added Ryan's legend patch - JDH


2006-1-12 Fixed numpy / numeric to use .dtype.char to keep in SYNC with numpy SVN

---------------------------

2006-1-11 Released 0.86.1

2006-1-11 Fixed setup.py for win32 build and added rc template to the MANIFEST.in

2006-1-10 Added xpdf distiller option. matplotlibrc ps.usedistiller can now be
            none, false, ghostscript, or xpdf. Validation checks for
            dependencies. This needs testing, but the xpdf option should produce
            the highest-quality output and small file sizes - DSD

2006-01-10 For the usetex option, backend_ps now does all the LaTeX work in the
            os's temp directory - DSD

2006-1-10 Added checks for usetex dependencies. - DSD

---------------------------------

2006-1-9 Released 0.86

2006-1-4 Changed to support numpy (new name for scipy_core) - TEO

2006-1-4 Added Mark's scaled axes patch for shared axis

2005-12-28 Added Chris Barker's build_wxagg patch - JDH

2005-12-27 Altered numerix/scipy to support new scipy package
           structure - TEO

2005-12-20 Fixed Jame's Boyles date tick reversal problem - JDH

2005-12-20 Added Jouni's rc patch to support lists of keys to set on -
           JDH

2005-12-12 Updated pyparsing and mathtext for some speed enhancements
           (Thanks Paul McGuire) and minor fixes to scipy numerix and
           setuptools

2005-12-12 Matplotlib data is now installed as package_data in
           the matplotlib module.  This gets rid of checking the
           many possibilities in matplotlib._get_data_path() - CM

2005-12-11 Support for setuptools/pkg_resources to build and use
           matplotlib as an egg. Still allows matplotlib to exist
           using a traditional distutils install. - ADS

2005-12-03 Modified setup to build matplotlibrc based on compile time
           findings.  It will set numerix in the order of scipy,
           numarray, Numeric depending on which are founds, and
           backend as in preference order GTKAgg, WXAgg, TkAgg, GTK,
           Agg, PS

2005-12-03 Modified scipy patch to support Numeric, scipy and numarray
           Some work remains to be done because some of the scipy
           imports are broken if only the core is installed.  e.g.,
           apparently we need from scipy.basic.fftpack import * rather
           than from scipy.fftpack import *

2005-12-03 Applied some fixes to Nicholas Young's nonuniform image
           patch

2005-12-01 Applied Alex Gontmakher hatch patch - PS only for now

2005-11-30 Added Rob McMullen's EMF patch

2005-11-30 Added Daishi's patch for scipy

2005-11-30 Fixed out of bounds draw markers segfault in agg

2005-11-28 Got TkAgg blitting working 100% (cross fingers) correctly. - CM

2005-11-27 Multiple changes in cm.py, colors.py, figure.py, image.py,
           contour.py, contour_demo.py; new _cm.py, examples/image_masked.py.
           1) Separated the color table data from cm.py out into
           a new file, _cm.py, to make it easier to find the actual
           code in cm.py and to add new colormaps.  Also added
           some line breaks to the color data dictionaries. Everything
           from _cm.py is imported by cm.py, so the split should be
           transparent.
           2) Enabled automatic generation of a colormap from
           a list of colors in contour; see modified
           examples/contour_demo.py.
           3) Support for imshow of a masked array, with the
           ability to specify colors (or no color at all) for
           masked regions, and for regions that are above or
           below the normally mapped region.  See
           examples/image_masked.py.
           4) In support of the above, added two new classes,
           ListedColormap, and no_norm, to colors.py, and modified
           the Colormap class to include common functionality. Added
           a clip kwarg to the normalize class.  Reworked color
           handling in contour.py, especially in the ContourLabeller
           mixin.
           - EF

2005-11-25 Changed text.py to ensure color is hashable. EF

--------------------------------

2005-11-16 Released 0.85

2005-11-16 Changed the default default linewidth in rc to 1.0

2005-11-16 Replaced agg_to_gtk_drawable with pure pygtk pixbuf code in
           backend_gtkagg.  When the equivalent is doe for blit, the
           agg extension code will no longer be needed

2005-11-16 Added a maxdict item to cbook to prevent caches from
           growing w/o bounds

2005-11-15 Fixed a colorup/colordown reversal bug in finance.py --
           Thanks Gilles

2005-11-15 Applied Jouni K Steppanen's boxplot patch SF patch#1349997
           - JDH


2005-11-09 added axisbelow attr for Axes to determine whether ticks and such
             are above or below the actors

2005-11-08 Added Nicolas' irregularly spaced image patch


2005-11-08 Deprecated HorizontalSpanSelector and replaced with
           SpanSelection that takes a third arg, direction.  The
           new SpanSelector supports horizontal and vertical span
           selection, and the appropriate min/max is returned. - CM

2005-11-08 Added lineprops dialog for gtk

2005-11-03 Added FIFOBuffer class to mlab to support real time feeds
           and examples/fifo_buffer.py

2005-11-01 Contributed Nickolas Young's patch for afm mathtext to
           support mathtext based upon the standard postscript Symbol
           font when ps.usetex = True.

2005-10-26 Added support for scatter legends - thanks John Gill

2005-10-20 Fixed image clipping bug that made some tex labels
           disappear.   JDH

2005-10-14 Removed sqrt from dvipng 1.6 alpha channel mask.

2005-10-14 Added width kwarg to hist function

2005-10-10 Replaced all instances of os.rename with shutil.move

2005-10-05 Added Michael Brady's ydate patch

2005-10-04 Added rkern's texmanager patch

2005-09-25 contour.py modified to use a single ContourSet class
           that handles filled contours, line contours, and labels;
           added keyword arg (clip_ends) to contourf.
           Colorbar modified to work with new ContourSet object;
           if the ContourSet has lines rather than polygons, the
           colorbar will follow suit. Fixed a bug introduced in
           0.84, in which contourf(...,colors=...) was broken - EF

-------------------------------

2005-09-19 Released 0.84

2005-09-14 Added a new 'resize_event' which triggers a callback with a
           backend_bases.ResizeEvent object - JDH

2005-09-14 font_manager.py: removed chkfontpath from x11FontDirectory() - SC

2005-09-14 Factored out auto date locator/formatter factory code into
           matplotlib.date.date_ticker_factory; applies John Bryne's
           quiver patch.

2005-09-13 Added Mark's axes positions history patch #1286915

2005-09-09 Added support for auto canvas resizing with
           fig.set_figsize_inches(9,5,forward=True) # inches
           OR
           fig.resize(400,300)  # pixels

2005-09-07 figure.py: update Figure.draw() to use the updated
           renderer.draw_image() so that examples/figimage_demo.py works again.
           examples/stock_demo.py: remove data_clipping (which no longer
           exists) - SC

2005-09-06 Added Eric's tick.direction patch: in or out in rc

2005-09-06 Added Martin's rectangle selector widget

2005-09-04 Fixed a logic err in text.py that was preventing rgxsuper
           from matching - JDH

2005-08-29 Committed Ken's wx blit patch #1275002

2005-08-26 colorbar modifications - now uses contourf instead of imshow
           so that colors used by contourf are displayed correctly.
           Added two new keyword args (cspacing and clabels) that are
           only relevant for ContourMappable images - JSWHIT

2005-08-24 Fixed a PS image bug reported by Darren - JDH

2005-08-23 colors.py: change hex2color() to accept unicode strings as well as
           normal strings. Use isinstance() instead of types.IntType etc - SC

2005-08-16 removed data_clipping line and rc property - JDH

2005-08-22 backend_svg.py: Remove redundant "x=0.0 y=0.0" from svg element.
           Increase svg version from 1.0 to 1.1. Add viewBox attribute to svg
           element to allow SVG documents to scale-to-fit into an arbitrary
           viewport - SC

2005-08-16 Added Eric's dot marker patch - JDH

2005-08-08 Added blitting/animation for TkAgg - CM

2005-08-05 Fixed duplicate tickline bug - JDH

2005-08-05 Fixed a GTK animation bug that cropped up when doing
           animations in gtk//gtkagg canvases that had widgets packed
           above them

2005-08-05 Added Clovis Goldemberg patch to the tk save dialog

2005-08-04 Removed origin kwarg from backend.draw_image.  origin is
           handled entirely by the frontend now.

2005-07-03 Fixed a bug related to TeX commands in backend_ps

2005-08-03 Fixed SVG images to respect upper and lower origins.

2005-08-03 Added flipud method to image and removed it from to_str.

2005-07-29 Modified figure.figaspect to take an array or number;
           modified backend_svg to write utf-8 - JDH

2005-07-30 backend_svg.py: embed png image files in svg rather than linking
           to a separate png file, fixes bug #1245306 (thanks to Norbert Nemec
           for the patch) - SC

---------------------------

2005-07-29 Released 0.83.2

2005-07-27 Applied SF patch 1242648: minor rounding error in
           IndexDateFormatter in dates.py

2005-07-27 Applied sf patch 1244732: Scale axis such that circle
           looks like circle - JDH

2005-07-29 Improved message reporting in texmanager and backend_ps - DSD

2005-07-28 backend_gtk.py: update FigureCanvasGTK.draw() (needed due to the
           recent expose_event() change) so that examples/anim.py works in the
           usual way - SC

2005-07-26 Added new widgets Cursor and HorizontalSpanSelector to
           matplotlib.widgets.  See examples/widgets/cursor.py and
           examples/widgets/span_selector.py - JDH

2005-07-26 added draw event to mpl event hierarchy -- triggered on
           figure.draw

2005-07-26 backend_gtk.py: allow 'f' key to toggle window fullscreen mode

2005-07-26 backend_svg.py: write "<.../>" elements all on one line and remove
           surplus spaces - SC

2005-07-25 backend_svg.py: simplify code by deleting GraphicsContextSVG and
           RendererSVG.new_gc(), and moving the gc.get_capstyle() code into
           RendererSVG._get_gc_props_svg() - SC

2005-07-24 backend_gtk.py: call FigureCanvasBase.motion_notify_event() on
           all motion-notify-events, not just ones where a modifier key or
           button has been pressed (fixes bug report from Niklas Volbers) - SC

2005-07-24 backend_gtk.py: modify print_figure() use own pixmap, fixing
           problems where print_figure() overwrites the display pixmap.
           return False from all button/key etc events - to allow the event
           to propagate further - SC

2005-07-23 backend_gtk.py: change expose_event from using set_back_pixmap();
           clear() to draw_drawable() - SC

2005-07-23 backend_gtk.py: removed pygtk.require()
           matplotlib/__init__.py: delete 'FROZEN' and 'McPLError' which are
           no longer used - SC

2005-07-22 backend_gdk.py: removed pygtk.require() - SC

2005-07-21 backend_svg.py: Remove unused imports. Remove methods doc strings
           which just duplicate the docs from backend_bases.py. Rename
           draw_mathtext to _draw_mathtext. - SC

2005-07-17 examples/embedding_in_gtk3.py: new example demonstrating placing
           a FigureCanvas in a gtk.ScrolledWindow - SC

2005-07-14 Fixed a Windows related bug (#1238412) in texmanager - DSD

2005-07-11 Fixed color kwarg bug, setting color=1 or 0 caused an
                   exception - DSD

2005-07-07 Added Eric's MA set_xdata Line2D fix - JDH

2005-07-06 Made HOME/.matplotlib the new config dir where the
           matplotlibrc file, the ttf.cache, and the tex.cache live.
           The new default filenames in .matplotlib have no leading
           dot and are not hidden.  e.g., the new names are matplotlibrc
           tex.cache ttffont.cache.  This is how ipython does it so it
           must be right.  If old files are found, a warning is issued
           and they are moved to the new location.  Also fixed
           texmanager to put all files, including temp files in
           ~/.matplotlib/tex.cache, which allows you to usetex in
           non-writable dirs.

2005-07-05 Fixed bug #1231611 in subplots adjust layout.  The problem
           was that the text cacheing mechanism was not using the
           transformation affine in the key. - JDH

2005-07-05 Fixed default backend import problem when using API (SF bug
           # 1209354 -  see API_CHANGES for more info - JDH

2005-07-04 backend_gtk.py: require PyGTK version 2.0.0 or higher - SC

2005-06-30 setupext.py: added numarray_inc_dirs for building against
           numarray when not installed in standard location - ADS

2005-06-27 backend_svg.py: write figure width, height as int, not float.
           Update to fix some of the pychecker warnings - SC

2005-06-23 Updated examples/agg_test.py to demonstrate curved paths
           and fills - JDH

2005-06-21 Moved some texmanager and backend_agg tex caching to class
           level rather than instance level - JDH

2005-06-20 setupext.py: fix problem where _nc_backend_gdk is installed to the
           wrong directory - SC

2005-06-19 Added 10.4 support for CocoaAgg. - CM

2005-06-18 Move Figure.get_width_height() to FigureCanvasBase and return
           int instead of float. - SC

2005-06-18 Applied Ted Drain's QtAgg patch: 1) Changed the toolbar to
           be a horizontal bar of push buttons instead of a QToolbar
           and updated the layout algorithms in the main window
           accordingly.  This eliminates the ability to drag and drop
           the toolbar and detach it from the window.  2) Updated the
           resize algorithm in the main window to show the correct
           size for the plot widget as requested.  This works almost
           correctly right now.  It looks to me like the final size of
           the widget is off by the border of the main window but I
           haven't figured out a way to get that information yet.  We
           could just add a small margin to the new size but that
           seems a little hacky.  3) Changed the x/y location label to
           be in the toolbar like the Tk backend instead of as a
           status line at the bottom of the widget.  4) Changed the
           toolbar pixmaps to use the ppm files instead of the png
           files.  I noticed that the Tk backend buttons looked much
           nicer and it uses the ppm files so I switched them.

2005-06-17 Modified the gtk backend to not queue mouse motion events.
           This allows for live updates when dragging a slider. - CM

2005-06-17 Added starter CocoaAgg backend.  Only works on OS 10.3 for
           now and requires PyObjC.  (10.4 is high priority) - CM

2005-06-17 Upgraded pyparsing and applied Paul McGuire's suggestions
           for speeding things up.  This more than doubles the speed
           of mathtext in my simple tests. JDH

2005-06-16 Applied David Cooke's subplot make_key patch

----------------------------------

2005-06-15 0.82 released

2005-06-15 Added subplot config tool to GTK* backends -- note you must
           now import the NavigationToolbar2 from your backend of
           choice rather than from backend_gtk because it needs to
           know about the backend specific canvas -- see
           examples/embedding_in_gtk2.py.  Ditto for wx backend -- see
           examples/embedding_in_wxagg.py

2005-06-15 backend_cairo.py: updated to use pycairo 0.5.0 - SC

2005-06-14 Wrote some GUI neutral widgets (Button, Slider,
           RadioButtons, CheckButtons) in matplotlib.widgets.  See
           examples/widgets/\*.py - JDH

2005-06-14 Exposed subplot parameters as rc vars and as the fig
           SubplotParams instance subplotpars.  See
           figure.SubplotParams, figure.Figure.subplots_adjust and the
           pylab method subplots_adjust and
           examples/subplots_adjust.py .  Also added a GUI neutral
           widget for adjusting subplots, see
           examples/subplot_toolbar.py - JDH

2005-06-13 Exposed cap and join style for lines with new rc params and
           line properties

        lines.dash_joinstyle : miter        # miter|round|bevel
        lines.dash_capstyle : butt          # butt|round|projecting
        lines.solid_joinstyle : miter       # miter|round|bevel
        lines.solid_capstyle : projecting   # butt|round|projecting


2005-06-13 Added kwargs to Axes init

2005-06-13 Applied Baptiste's tick patch - JDH

2005-06-13 Fixed rc alias 'l' bug reported by Fernando by removing
           aliases for mainlevel rc options. - JDH

2005-06-10 Fixed bug #1217637 in ticker.py - DSD

2005-06-07 Fixed a bug in texmanager.py: .aux files not being removed - DSD

2005-06-08 Added Sean Richard's hist binning fix -- see API_CHANGES - JDH

2005-06-07 Fixed a bug in texmanager.py: .aux files not being removed
           - DSD


----------------------

2005-06-07 matplotlib-0.81 released

2005-06-06 Added autoscale_on prop to axes

2005-06-06 Added Nick's picker "among" patch - JDH

2005-06-05 Fixed a TeX/LaTeX font discrepency in backend_ps. - DSD

2005-06-05 Added a ps.distill option in rc settings. If True, postscript
           output will be distilled using ghostscript, which should trim
           the file size and allow it to load more quickly. Hopefully this
           will address the issue of large ps files due to font
           definitions. Tested with gnu-ghostscript-8.16. - DSD

2005-06-03 Improved support for tex handling of text in backend_ps. - DSD

2005-06-03 Added rc options to render text with tex or latex, and to select
           the latex font package. - DSD

2005-06-03 Fixed a bug in ticker.py causing a ZeroDivisionError

2005-06-02 backend_gtk.py remove DBL_BUFFER, add line to expose_event to
           try to fix pygtk 2.6 redraw problem - SC

2005-06-01 The default behavior of ScalarFormatter now renders scientific
           notation and large numerical offsets in a label at the end of
           the axis. - DSD

2005-06-01 Added Nicholas' frombyte image patch - JDH

2005-05-31 Added vertical TeX support for agg - JDH

2005-05-31 Applied Eric's cntr patch - JDH

2005-05-27 Finally found the pesky agg bug (which Maxim was kind
           enough to fix within hours) that was causing a segfault in
           the win32 cached marker drawing.  Now windows users can get
           the enormouse performance benefits of caced markers w/o
           those occasional pesy screenshots. - JDH

2005-05-27 Got win32 build system working again, using a more recent
           version of gtk and pygtk in the win32 build, gtk 2.6 from
           http://www.gimp.org/~tml/gimp/win32/downloads.html (you
           will also need libpng12.dll to use these).  I haven't
           tested whether this binary build of mpl for win32 will work
           with older gtk runtimes, so you may need to upgrade.

2005-05-27 Fixed bug where 2nd wxapp could be started if using wxagg
           backend. - ADS

2005-05-26 Added Daishi text with dash patch -- see examples/dashtick.py

2005-05-26 Moved backend_latex functionality into backend_ps. If
           text.usetex=True, the PostScript backend will use LaTeX to
           generate the .ps or .eps file. Ghostscript is required for
           eps output. - DSD

2005-05-24 Fixed alignment and color issues in latex backend. - DSD

2005-05-21 Fixed raster problem for small rasters with dvipng -- looks
           like it was a premultipled alpha problem - JDH

2005-05-20 Added linewidth and faceted kwarg to scatter to control
           edgewidth and color.  Also added autolegend patch to
           inspect line segments.

2005-05-18 Added Orsay and JPL qt fixes - JDH

2005-05-17 Added a psfrag latex backend -- some alignment issues need
           to be worked out. Run with -dLaTeX and a *.tex file and
           *.eps file are generated.  latex and dvips the generated
           latex file to get ps output.  Note xdvi *does* not work,
           you must generate ps.- JDH

2005-05-13 Added Florent Rougon's Axis set_label1
           patch

2005-05-17 pcolor optimization, fixed bug in previous pcolor patch - JSWHIT

2005-05-16 Added support for masked arrays in pcolor - JSWHIT


2005-05-12 Started work on TeX text for antigrain using pngdvi -- see
           examples/tex_demo.py and the new module
           matplotlib.texmanager.  Rotated text not supported and
           rendering small glyps is not working right yet.  BUt large
           fontsizes and/or high dpi saved figs work great.

2005-05-10 New image resize options interpolation options.  New values
           for the interp kwarg are

           'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
           'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
           'lanczos', 'blackman'

           See help(imshow) for details, particularly the
           interpolation, filternorm and filterrad kwargs


2005-05-10 Applied Eric's contour mem leak fixes - JDH

2005-05-10 Extended python agg wrapper and started implementing
           backend_agg2, an agg renderer based on the python wrapper.
           This will be more flexible and easier to extend than the
           current backend_agg.  See also examples/agg_test.py - JDH

2005-05-09 Added Marcin's no legend patch to exclude lines from the
           autolegend builder

           plot(x, y, label='nolegend')

2005-05-05 Upgraded to agg23

2005-05-05 Added newscalarformatter_demo.py to examples. -DSD

2005-05-04 Added NewScalarFormatter. Improved formatting of ticklabels,
           scientific notation, and the ability to plot large large
           numbers with small ranges, by determining a numerical offset.
           See ticker.NewScalarFormatter for more details. -DSD

2005-05-03 Added the option to specify a delimiter in pylab.load -DSD

2005-04-28 Added Darren's line collection example

2005-04-28 Fixed aa property in agg - JDH

2005-04-27 Set postscript page size in .matplotlibrc - DSD

2005-04-26 Added embedding in qt example. - JDH

2005-04-14 Applied Michael Brady's qt backend patch: 1) fix a bug
           where keyboard input was grabbed by the figure and not
           released  2) turn on cursor changes  3) clean up a typo
           and commented-out print statement. - JDH


2005-04-14 Applied Eric Firing's masked data lines patch and contour
           patch.  Support for masked arrays has been added to the
           plot command and to the Line2D object.  Only the valid
           points are plotted.  A "valid_only" kwarg was added to the
           get_xdata() and get_ydata() methods of Line2D; by default
           it is False, so that the original data arrays are
           returned. Setting it to True returns the plottable points.
           - see examples/masked_demo.py - JDH

2005-04-13 Applied Tim Leslie's arrow key event handling patch - JDH


---------------------------

0.80 released

2005-04-11 Applied a variant of rick's xlim/ylim/axis patch.  These
           functions now take kwargs to let you selectively alter only
           the min or max if desired.  e.g., xlim(xmin=2) or
           axis(ymax=3).  They always return the new lim. - JDH


2005-04-11 Incorporated Werner's wx patch -- wx backend should be
           compatible with wxpython2.4 and recent versions of 2.5.
           Some early versions of wxpython 2.5 will not work because
           there was a temporary change in the dc API that was rolled
           back to make it 2.4 compliant

2005-04-11 modified tkagg show so that new figure window pops up on
           call to figure

2005-04-11 fixed wxapp init bug

2005-04-02 updated backend_ps.draw_lines, draw_markers for use with the
           new API - DSD

2005-04-01 Added editable polygon example

------------------------------

2005-03-31 0.74 released

2005-03-30 Fixed and added checks for floating point inaccuracy in
           ticker.Base - DSD

2005-03-30 updated /ellipse definition in backend_ps.py to address bug
           #1122041 - DSD

2005-03-29 Added unicode support for Agg and PS - JDH

2005-03-28 Added Jarrod's svg patch for text - JDH

2005-03-28 Added Ludal's arrow and quiver patch - JDH

2005-03-28 Added label kwarg to Axes to facilitate forcing the
           creation of new Axes with otherwise identical attributes

2005-03-28 Applied boxplot and OSX font search patches

2005-03-27 Added ft2font NULL check to fix Japanase font bug - JDH

2005-03-27 Added sprint legend patch plus John Gill's tests and fix --
           see examples/legend_auto.py  - JDH

---------------------------

2005-03-19 0.73.1 released

2005-03-19 Reverted wxapp handling because it crashed win32 - JDH

2005-03-18 Add .number attribute to figure objects returned by figure() - FP

---------------------------

2005-03-18 0.73 released

2005-03-16 Fixed labelsep bug

2005-03-16 Applied Darren's ticker fix for small ranges - JDH

2005-03-16 Fixed tick on horiz colorbar - JDH

2005-03-16 Added Japanses winreg patch - JDH

2005-03-15 backend_gtkagg.py: changed to use double buffering, this fixes
           the problem reported Joachim Berdal Haga - "Parts of plot lagging
           from previous frame in animation". Tested with anim.py and it makes
           no noticable difference to performance (23.7 before, 23.6 after)
           - SC

2005-03-14 add src/_backend_gdk.c extension to provide a substitute function
           for pixbuf.get_pixels_array(). Currently pixbuf.get_pixels_array()
           only works with Numeric, and then only works if pygtk has been
           compiled with Numeric support. The change provides a function
           pixbuf_get_pixels_array() which works with Numeric and numarray and
           is always available. It means that backend_gtk should be able to
           display images and mathtext in all circumstances. - SC

2005-03-11 Upgraded CXX to 5.3.1

2005-03-10 remove GraphicsContextPS.set_linestyle()
           and GraphicsContextSVG.set_linestyle() since they do no more than
           the base class GraphicsContext.set_linestyle() - SC

2005-03-09 Refactored contour functionality into dedicated module

2005-03-09 Added Eric's contourf updates and Nadia's clabel functionality

2005-03-09 Moved colorbar to figure.Figure to expose it for API developers
          - JDH

2005-03-09 backend_cairo.py: implemented draw_markers() - SC

2005-03-09 cbook.py: only use enumerate() (the python version) if the builtin
       version is not available.
           Add new function 'izip' which is set to itertools.izip if available
           and the python equivalent if not available. - SC

2005-03-07 backend_gdk.py: remove PIXELS_PER_INCH from points_to_pixels(), but
       still use it to adjust font sizes. This allows the GTK version of
           line_styles.py to more closely match GTKAgg, previously the markers
           were being drawn too large. - SC

2005-03-01 Added Eric's contourf routines

2005-03-01 Added start of proper agg SWIG wrapper.  I would like to
           expose agg functionality directly a the user level and this
           module will serve that purpose eventually, and will
           hopefully take over most of the functionality of the
           current _image and _backend_agg modules.  - JDH

2005-02-28 Fixed polyfit / polyval to convert input args to float
           arrays - JDH


2005-02-25 Add experimental feature to backend_gtk.py to enable/disable
           double buffering (DBL_BUFFER=True/False) - SC

2005-02-24 colors.py change ColorConverter.to_rgb() so it always returns rgb
           (and not rgba), allow cnames keys to be cached, change the exception
           raised from RuntimeError to ValueError (like hex2color())
           hex2color() use a regular expression to check the color string is
           valid  - SC


2005-02-23 Added rc param ps.useafm so backend ps can use native afm
           fonts or truetype.  afme breaks mathtext but causes much
           smaller font sizes and may result in images that display
           better in some contexts (e.g., pdfs incorporated into latex
           docs viewed in acrobat reader).  I would like to extend
           this approach to allow the user to use truetype only for
           mathtext, which should be easy.

2005-02-23 Used sequence protocol rather than tuple in agg collection
           drawing routines for greater flexibility - JDH


--------------------------------

2005-02-22 0.72.1 released

2005-02-21 fixed linestyles for collections -- contour now dashes for
           levels <0

2005-02-21 fixed ps color bug - JDH

2005-02-15 fixed missing qt file

2005-02-15 banished error_msg and report_error.  Internal backend
           methods like error_msg_gtk are preserved.  backend writers,
           check your backends, and diff against 0.72 to make sure I
           did the right thing! - JDH


2005-02-14 Added enthought traits to matplotlib tree - JDH

------------------------

2005-02-14 0.72 released

2005-02-14 fix bug in cbook alltrue() and onetrue() - SC

2005-02-11 updated qtagg backend from Ted - JDH

2005-02-11 matshow fixes for figure numbering, return value and docs - FP

2005-02-09 new zorder example for fine control in zorder_demo.py - FP

2005-02-09 backend renderer draw_lines now has transform in backend,
           as in draw_markers; use numerix in _backend_agg, aded small
           line optimization to agg

2005-02-09 subplot now deletes axes that it overlaps

2005-02-08 Added transparent support for gzipped files in load/save - Fernando
           Perez (FP from now on).

2005-02-08 Small optimizations in PS backend.  They may have a big impact for
           large plots, otherwise they don't hurt  - FP

2005-02-08 Added transparent support for gzipped files in load/save - Fernando
           Perez (FP from now on).

2005-02-07 Added newstyle path drawing for markers - only implemented
           in agg currently - JDH

2005-02-05 Some superscript text optimizations for ticking log plots

2005-02-05 Added some default key press events to pylab figures: 'g'
           toggles grid - JDH

2005-02-05 Added some support for handling log switching for lines
           that have nonpos data - JDH

2005-02-04 Added Nadia's contour patch - contour now has matlab
           compatible syntax; this also fixed an unequal sized contour
           array bug- JDH

2005-02-04 Modified GTK backends to allow the FigureCanvas to be resized
       smaller than its original size - SC

2005-02-02 Fixed a bug in dates mx2num - JDH

2005-02-02 Incorporated Fernando's matshow - JDH

2005-02-01 Added Fernando's figure num patch, including experemental
           support for pylab backend switching, LineCOllection.color
           warns, savefig now a figure method, fixed a close(fig) bug
           - JDH

2005-01-31 updated datalim in contour - JDH

2005-01-30 Added backend_qtagg.py provided by Sigve Tjora - SC

2005-01-28 Added tk.inspect rc param to .matplotlibrc.  IDLE users
           should set tk.pythoninspect:True and interactive:True and
           backend:TkAgg

2005-01-28 Replaced examples/interactive.py with an updated script from
           Fernando Perez - SC

2005-01-27 Added support for shared x or y axes.  See
           examples/shared_axis_demo.py and examples/ganged_plots.py

2005-01-27 Added Lee's patch for missing symbols \leq and \LEFTbracket
           to _mathtext_data - JDH

2005-01-26 Added Baptiste's two scales patch -- see help(twinx) in the
           pylab interface for more info.  See also
           examples/two_scales.py

2005-01-24 Fixed a mathtext parser bug that prevented font changes in
           sub/superscripts - JDH

2005-01-24 Fixed contour to work w/ interactive changes in colormaps,
           clim, etc - JDH

-----------------------------

2005-01-21 matplotlib-0.71 released

2005-01-21 Refactored numerix to solve vexing namespace issues - JDH

2005-01-21 Applied Nadia's contour bug fix - JDH

2005-01-20 Made some changes to the contour routine - particularly
           region=1 seems t fix a lot of the zigzag strangeness.
           Added colormaps as default for contour - JDH

2005-01-19 Restored builtin names which were overridden (min, max,
           abs, round, and sum) in pylab.  This is a potentially
           significant change for those who were relying on an array
           version of those functions that previously overrode builtin
           function names. - ADS

2005-01-18 Added accents to mathtext: \hat, \breve, \grave, \bar,
           \acute, \tilde, \vec, \dot, \ddot.  All of them have the
           same syntax, e.g., to make an overbar you do \bar{o} or to
           make an o umlaut you do \ddot{o}. The shortcuts are also
           provided, e.g., \"o \'e \`e \~n \.x \^y - JDH

2005-01-18 Plugged image resize memory leaks - JDH

2005-01-18 Fixed some mathtext parser problems relating to superscripts

2005-01-17 Fixed a yticklabel problem for colorbars under change of
           clim - JDH

2005-01-17 Cleaned up Destroy handling in wx reducing memleak/fig from
           approx 800k to approx 6k- JDH

2005-01-17 Added kappa to latex_to_bakoma - JDH

2005-01-15 Support arbitrary colorbar axes and horizontal colorbars - JDH

2005-01-15 Fixed colormap number of colors bug so that the colorbar
           has the same discretization as the image - JDH

2005-01-15 Added Nadia's x,y contour fix - JDH

2005-01-15 backend_cairo: added PDF support which requires pycairo 0.1.4.
           Its not usable yet, but is ready for when the Cairo PDF backend
           matures - SC

2005-01-15 Added Nadia's x,y contour fix

2005-01-12 Fixed set clip_on bug in artist - JDH

2005-01-11 Reverted pythoninspect in tkagg - JDH

2005-01-09 Fixed a backend_bases event bug caused when an event is
           triggered when location is None - JDH

2005-01-07 Add patch from Stephen Walton to fix bug in pylab.load()
           when the % character is included in a comment. - ADS

2005-01-07 Added markerscale attribute to Legend class.  This allows
           the marker size in the legend to be adjusted relative to
           that in the plot. - ADS

2005-01-06 Add patch from Ben Vanhaeren to make the FigureManagerGTK vbox a
           public attribute - SC

----------------------------

2004-12-30 Release 0.70

2004-12-28 Added coord location to key press and added a
           examples/picker_demo.py

2004-12-28 Fixed coords notification in wx toolbar - JDH

2004-12-28 Moved connection and disconnection event handling to the
           FigureCanvasBase.  Backends now only need to connect one
           time for each of the button press, button release and key
           press/release functions.  The base class deals with
           callbacks and multiple connections.  This fixes flakiness
           on some backends (tk, wx) in the presence of multiple
           connections and/or disconnect - JDH

2004-12-27 Fixed PS mathtext bug where color was not set - Jochen
           please verify correct - JDH

2004-12-27 Added Shadow class and added shadow kwarg to legend and pie
           for shadow effect - JDH

2004-12-27 Added pie charts and new example/pie_demo.py

2004-12-23 Fixed an agg text rotation alignment bug, fixed some text
           kwarg processing bugs, and added examples/text_rotation.py
           to explain and demonstrate how text rotations and alignment
           work in matplotlib. - JDH

-----------------------

2004-12-22 0.65.1 released - JDH

2004-12-22 Fixed colorbar bug which caused colorbar not to respond to
           changes in colormap in some instances - JDH

2004-12-22 Refactored NavigationToolbar in tkagg to support app
           embedding , init now takes (canvas, window) rather than
           (canvas, figman) - JDH

2004-12-21 Refactored axes and subplot management - removed
           add_subplot and add_axes from the FigureManager.  classic
           toolbar updates are done via an observer pattern on the
           figure using add_axobserver.  Figure now maintains the axes
           stack (for gca) and supports axes deletion.  Ported changes
           to GTK, Tk, Wx, and FLTK.  Please test!  Added delaxes - JDH

2004-12-21 Lots of image optimizations - 4x performance boost over
           0.65 JDH

2004-12-20 Fixed a figimage bug where the axes is shown and modified
           tkagg to move the destroy binding into the show method.

2004-12-18 Minor refactoring of NavigationToolbar2 to support
           embedding in an application - JDH

2004-12-14 Added linestyle to collections (currently broken) -  JDH

2004-12-14 Applied Nadia's setupext patch to fix libstdc++ link
           problem with contour and solaris  -JDH

2004-12-14 A number of pychecker inspired fixes, including removal of
           True and False from cbook which I erroneously thought was
           needed for python2.2 - JDH

2004-12-14 Finished porting doc strings for set introspection.
           Used silent_list for many get funcs that return
           lists. JDH

2004-12-13 dates.py: removed all timezone() calls, except for UTC - SC

----------------------------

2004-12-13 0.65 released - JDH

2004-12-13 colors.py: rgb2hex(), hex2color() made simpler (and faster), also
           rgb2hex() - added round() instead of integer truncation
           hex2color() - changed 256.0 divisor to 255.0, so now
           '#ffffff' becomes (1.0,1.0,1.0) not (0.996,0.996,0.996) - SC

2004-12-11 Added ion and ioff to pylab interface - JDH

2004-12-11 backend_template.py: delete FigureCanvasTemplate.realize() - most
           backends don't use it and its no longer needed

           backend_ps.py, backend_svg.py: delete show() and
           draw_if_interactive() - they are not needed for image backends

           backend_svg.py: write direct to file instead of StringIO
           - SC

2004-12-10 Added zorder to artists to control drawing order of lines,
           patches and text in axes.  See examples/zoder_demo.py - JDH

2004-12-10 Fixed colorbar bug with scatter - JDH

2004-12-10 Added Nadia Dencheva <dencheva@stsci.edu> contour code - JDH

2004-12-10 backend_cairo.py: got mathtext working - SC

2004-12-09 Added Norm Peterson's svg clipping patch

2004-12-09 Added Matthew Newville's wx printing patch

2004-12-09 Migrated matlab to pylab - JDH

2004-12-09 backend_gtk.py: split into two parts
           - backend_gdk.py - an image backend
           - backend_gtk.py - A GUI backend that uses GDK - SC

2004-12-08 backend_gtk.py: remove quit_after_print_xvfb(\*args), show_xvfb(),
           Dialog_MeasureTool(gtk.Dialog) one month after sending mail to
           matplotlib-users asking if anyone still uses these functions - SC

2004-12-02 backend_bases.py, backend_template.py: updated some of the method
           documentation to make them consistent with each other - SC

2004-12-04 Fixed multiple bindings per event for TkAgg mpl_connect and
           mpl_disconnect.  Added a "test_disconnect" command line
           parameter to coords_demo.py  JTM

2004-12-04 Fixed some legend bugs JDH

2004-11-30 Added over command for oneoff over plots.  e.g., over(plot, x,
           y, lw=2).  Works with any plot function.

2004-11-30 Added bbox property to text - JDH

2004-11-29 Zoom to rect now respect reversed axes limits (for both
           linear and log axes). - GL

2004-11-29 Added the over command to the matlab interface.  over
           allows you to add an overlay plot regardless of hold
           state. - JDH

2004-11-25 Added Printf to mplutils for printf style format string
           formatting in C++ (should help write better exceptions)

2004-11-24 IMAGE_FORMAT: remove from agg and gtkagg backends as its no longer
           used - SC

2004-11-23 Added matplotlib compatible set and get introspection.  See
           set_and_get.py

2004-11-23 applied Norbert's patched and exposed legend configuration
           to kwargs - JDH

2004-11-23 backend_gtk.py: added a default exception handler - SC

2004-11-18 backend_gtk.py: change so that the backend knows about all image
           formats and does not need to use IMAGE_FORMAT in other backends - SC

2004-11-18 Fixed some report_error bugs in string interpolation as
           reported on SF bug tracker- JDH

2004-11-17 backend_gtkcairo.py: change so all print_figure() calls render using
           Cairo and get saved using backend_gtk.print_figure() - SC

2004-11-13 backend_cairo.py: Discovered the magic number (96) required for
           Cairo PS plots to come out the right size. Restored Cairo PS output
           and added support for landscape mode - SC

2004-11-13 Added ishold - JDH

2004-11-12 Added many new matlab colormaps - autumn bone cool copper
           flag gray hot hsv jet pink prism spring summer winter - PG

2004-11-11 greatly simplify the emitted postscript code - JV

2004-11-12 Added new plotting functions spy, spy2 for sparse matrix
           visualization - JDH

2004-11-11 Added rgrids, thetragrids for customizing the grid
           locations and labels for polar plots - JDH

2004-11-11 make the Gtk backends build without an X-server connection - JV

2004-11-10 matplotlib/__init__.py: Added FROZEN to signal we are running under
           py2exe (or similar) - is used by backend_gtk.py - SC

2004-11-09 backend_gtk.py: Made fix suggested by maffew@cat.org.au
           to prevent problems when py2exe calls pygtk.require(). - SC

2004-11-09 backend_cairo.py: Added support for printing to a fileobject.
           Disabled cairo PS output which is not working correctly. - SC

----------------------------------

2004-11-08 matplotlib-0.64 released

2004-11-04 Changed -dbackend processing to only use known backends, so
           we don't clobber other non-matplotlib uses of -d, like -debug.

2004-11-04 backend_agg.py: added IMAGE_FORMAT to list the formats that the
           backend can save to.
           backend_gtkagg.py: added support for saving JPG files by using the
           GTK backend - SC

2004-10-31 backend_cairo.py: now produces png and ps files (although the figure
           sizing needs some work). pycairo did not wrap all the necessary
           functions, so I wrapped them myself, they are included in the
           backend_cairo.py doc string. - SC

2004-10-31 backend_ps.py: clean up the generated PostScript code, use
           the PostScript stack to hold itermediate values instead of
           storing them in the dictionary. - JV

2004-10-30 backend_ps.py, ft2font.cpp, ft2font.h: fix the position of
           text in the PostScript output.  The new FT2Font method
           get_descent gives the distance between the lower edge of
           the bounding box and the baseline of a string.  In
           backend_ps the text is shifted upwards by this amount. - JV

2004-10-30 backend_ps.py: clean up the code a lot.  Change the
           PostScript output to be more DSC compliant.  All
           definitions for the generated PostScript are now in a
           PostScript dictionary 'mpldict'.  Moved the long comment
           about drawing ellipses from the PostScript output into a
           Python comment. - JV

2004-10-30 backend_gtk.py: removed FigureCanvasGTK.realize() as its no longer
           needed. Merged ColorManager into GraphicsContext
           backend_bases.py: For set_capstyle/joinstyle() only set cap or
           joinstyle if there is no error. - SC

2004-10-30 backend_gtk.py: tidied up print_figure() and removed some of the
           dependency on widget events - SC

2004-10-28 backend_cairo.py: The renderer is complete except for mathtext,
           draw_image() and clipping. gtkcairo works reasonably well. cairo
           does not yet create any files since I can't figure how to set the
           'target surface', I don't think pycairo wraps the required functions
           - SC

2004-10-28 backend_gtk.py: Improved the save dialog (GTK 2.4 only) so it
           presents the user with a menu of supported image formats - SC

2004-10-28 backend_svg.py: change print_figure() to restore original face/edge
           color
           backend_ps.py : change print_figure() to ensure original face/edge
           colors are restored even if there's an IOError - SC

2004-10-27 Applied Norbert's errorbar patch to support barsabove kwarg

2004-10-27 Applied Norbert's legend patch to support None handles

2004-10-27 Added two more backends: backend_cairo.py, backend_gtkcairo.py
           They are not complete yet, currently backend_gtkcairo just renders
           polygons, rectangles and lines - SC

2004-10-21 Added polar axes and plots - JDH

2004-10-20 Fixed corrcoef bug exposed by corrcoef(X) where X is matrix
           - JDH

2004-10-19 Added kwarg support to xticks and yticks to set ticklabel
           text properties -- thanks to T. Edward Whalen for the suggestion

2004-10-19 Added support for PIL images in imshow(), image.py - ADS

2004-10-19 Re-worked exception handling in _image.py and _transforms.py
           to avoid masking problems with shared libraries.  - JTM

2004-10-16 Streamlined the matlab interface wrapper, removed the
           noplot option to hist - just use mlab.hist instead.

2004-09-30 Added Andrew Dalke's strftime code to extend the range of
           dates supported by the DateFormatter - JDH

2004-09-30 Added barh - JDH

2004-09-30 Removed fallback to alternate array package from numerix
           so that ImportErrors are easier to debug.           JTM

2004-09-30 Add GTK+ 2.4 support for the message in the toolbar. SC

2004-09-30 Made some changes to support python22 - lots of doc
           fixes. - JDH

2004-09-29 Added a Verbose class for reporting - JDH

------------------------------------

2004-09-28 Released 0.63.0

2004-09-28 Added save to file object for agg - see
           examples/print_stdout.py

2004-09-24 Reorganized all py code to lib subdir

2004-09-24 Fixed axes resize image edge effects on interpolation -
           required upgrade to agg22 which fixed an agg bug related to
           this problem

2004-09-20 Added toolbar2 message display for backend_tkagg.  JTM


2004-09-17 Added coords formatter attributes.  These must be callable,
           and return a string for the x or y data. These will be used
           to format the x and y data for the coords box.  Default is
           the axis major formatter.  e.g.:

         # format the coords message box
         def price(x): return '$%1.2f'%x
         ax.format_xdata = DateFormatter('%Y-%m-%d')
         ax.format_ydata = price


2004-09-17 Total rewrite of dates handling to use python datetime with
           num2date, date2num and drange.  pytz for timezone handling,
           dateutils for spohisticated ticking.  date ranges from
           0001-9999 are supported.  rrules allow arbitrary date
           ticking.  examples/date_demo*.py converted to show new
           usage.  new example examples/date_demo_rrule.py shows how
           to use rrules in date plots.  The date locators are much
           more general and almost all of them have different
           constructors.  See matplotlib.dates for more info.

2004-09-15 Applied Fernando's backend __init__ patch to support easier
           backend maintenance.  Added his numutils to mlab.  JDH

2004-09-16 Re-designated all files in matplotlib/images as binary and
           w/o keyword substitution using "cvs admin -kb \*.svg ...".
           See binary files in "info cvs" under Linux.  This was messing
           up builds from CVS on windows since CVS was doing lf -> cr/lf
           and keyword substitution on the bitmaps.  - JTM

2004-09-15 Modified setup to build array-package-specific extensions
           for those extensions which are array-aware.  Setup builds
           extensions automatically for either Numeric, numarray, or
           both, depending on what you have installed.  Python proxy
           modules for the array-aware extensions import the version
           optimized for numarray or Numeric determined by numerix.
           - JTM

2004-09-15 Moved definitions of infinity from mlab to numerix to avoid
           divide by zero warnings for numarray - JTM

2004-09-09 Added axhline, axvline, axhspan and axvspan

-------------------------------

2004-08-30 matplotlib 0.62.4 released

2004-08-30 Fixed a multiple images with different extent bug,
           Fixed markerfacecolor as RGB tuple

2004-08-27 Mathtext now more than 5x faster.  Thanks to Paul Mcguire
           for fixes both to pyparsing and to the matplotlib grammar!
           mathtext broken on python2.2

2004-08-25 Exposed Darren's and Greg's log ticking and formatting
           options to semilogx and friends

2004-08-23 Fixed grid w/o args to toggle grid state - JDH

2004-08-11 Added Gregory's log patches for major and minor ticking

2004-08-18 Some pixel edge effects fixes for images

2004-08-18 Fixed TTF files reads in backend_ps on win32.

2004-08-18 Added base and subs properties for logscale plots, user
           modifiable using
           set_[x,y]scale('log',base=b,subs=[mt1,mt2,...]) - GL

2004-08-18 fixed a bug exposed by trying to find the HOME dir on win32
           thanks to Alan Issac for pointing to the light - JDH

2004-08-18 fixed errorbar bug in setting ecolor - JDH

2004-08-12 Added Darren Dale's exponential ticking patch

2004-08-11 Added Gregory's fltkagg backend

------------------------------

2004-08-09 matplotlib-0.61.0 released

2004-08-08 backend_gtk.py: get rid of the final PyGTK deprecation warning by
           replacing gtkOptionMenu with gtkMenu in the 2.4 version of the
           classic toolbar.

2004-08-06 Added Tk zoom to rect rectangle, proper idle drawing, and
           keybinding - JDH

2004-08-05 Updated installing.html and INSTALL - JDH

2004-08-01 backend_gtk.py: move all drawing code into the expose_event()

2004-07-28 Added Greg's toolbar2 and backend_*agg patches - JDH

2004-07-28 Added image.imread with support for loading png into
           numerix arrays

2004-07-28 Added key modifiers to events - implemented dynamic updates
           and rubber banding for interactive pan/zoom - JDH

2004-07-27 did a readthrough of SVG, replacing all the string
           additions with string interps for efficiency, fixed some
           layout problems, added font and image support (through
           external pngs) - JDH

2004-07-25 backend_gtk.py: modify toolbar2 to make it easier to support GTK+
           2.4. Add GTK+ 2.4 toolbar support. - SC

2004-07-24 backend_gtk.py: Simplified classic toolbar creation - SC

2004-07-24 Added images/matplotlib.svg to be used when GTK+ windows are
           minimised - SC

2004-07-22 Added right mouse click zoom for NavigationToolbar2 panning
       mode. - JTM

2004-07-22 Added NavigationToolbar2 support to backend_tkagg.
       Minor tweak to backend_bases.  - JTM

2004-07-22 Incorporated Gergory's renderer cache and buffer object
           cache - JDH

2004-07-22 Backend_gtk.py: Added support for GtkFileChooser, changed
           FileSelection/FileChooser so that only one instance pops up,
           and made them both modal. - SC

2004-07-21 Applied backend_agg memory leak patch from hayden -
           jocallo@online.no.  Found and fixed a leak in binary
           operations on transforms.  Moral of the story: never incref
           where you meant to decref!  Fixed several leaks in ft2font:
           moral of story: almost always return Py::asObject over
           Py::Object - JDH

2004-07-21 Fixed a to string memory allocation bug in agg and image
           modules - JDH

2004-07-21 Added mpl_connect and mpl_disconnect to matlab interface -
           JDH

2004-07-21 Added beginnings of users_guide to CVS - JDH

2004-07-20 ported toolbar2 to wx

2004-07-20 upgraded to agg21 - JDH

2004-07-20 Added new icons for toolbar2 - JDH

2004-07-19 Added vertical mathtext for \*Agg and GTK - thanks Jim
           Benson! - JDH

2004-07-16 Added ps/eps/svg savefig options to wx and gtk JDH

2004-07-15 Fixed python framework tk finder in setupext.py - JDH

2004-07-14 Fixed layer images demo which was broken by the 07/12 image
           extent fixes - JDH

2004-07-13 Modified line collections to handle arbitrary length
           segments for each line segment. - JDH

2004-07-13 Fixed problems with image extent and origin -
           set_image_extent deprecated.  Use imshow(blah, blah,
           extent=(xmin, xmax, ymin, ymax) instead  - JDH

2004-07-12 Added prototype for new nav bar with codifed event
           handling.  Use mpl_connect rather than connect for
           matplotlib event handling.  toolbar style determined by rc
           toolbar param.  backend status: gtk: prototype, wx: in
           progress, tk: not started - JDH

2004-07-11 backend_gtk.py: use builtin round() instead of redefining it.
           - SC

2004-07-10 Added embedding_in_wx3 example - ADS

2004-07-09 Added dynamic_image_wxagg to examples - ADS

2004-07-09 added support for embedding TrueType fonts in PS files - PEB

2004-07-09 fixed a sfnt bug exposed if font cache is not built

2004-07-09 added default arg None to matplotlib.matlab grid command to
           toggle current grid state

---------------------

2004-07-08 0.60.2 released

2004-07-08 fixed a mathtext bug for '6'

2004-07-08 added some numarray bug workarounds

--------------------------

2004-07-07 0.60 released

2004-07-07 Fixed a bug in dynamic_demo_wx


2004-07-07 backend_gtk.py: raise SystemExit immediately if
       'import pygtk' fails - SC

2004-07-05 Added new mathtext commands \over{sym1}{sym2} and
           \under{sym1}{sym2}

2004-07-05 Unified image and patch collections colormapping and
           scaling args.  Updated docstrings for all - JDH

2004-07-05 Fixed a figure legend bug and added
           examples/figlegend_demo.py - JDH

2004-07-01 Fixed a memory leak in image and agg to string methods

2004-06-25 Fixed fonts_demo spacing problems and added a kwargs
           version of the fonts_demo fonts_demo_kw.py - JDH

2004-06-25 finance.py: handle case when urlopen() fails - SC

2004-06-24 Support for multiple images on axes and figure, with
           blending.  Support for upper and lower image origins.
           clim, jet and gray functions in matlab interface operate on
           current image - JDH

2004-06-23 ported code to Perry's new colormap and norm scheme.  Added
           new rc attributes image.aspect, image.interpolation,
           image.cmap, image.lut, image.origin

2004-06-20 backend_gtk.py: replace gtk.TRUE/FALSE with True/False.
       simplified _make_axis_menu(). - SC

2004-06-19 anim_tk.py: Updated to use TkAgg by default (not GTK)
           backend_gtk_py: Added '_' in front of private widget
           creation functions - SC

2004-06-17 backend_gtk.py: Create a GC once in realise(), not every
           time draw() is called. - SC

2004-06-16 Added new py2exe FAQ entry and added frozen support in
           get_data_path for py2exe - JDH

2004-06-16 Removed GTKGD, which was always just a proof-of-concept
           backend - JDH

2004-06-16 backend_gtk.py updates to replace deprecated functions
       gtk.mainquit(), gtk.mainloop().
           Update NavigationToolbar to use the new GtkToolbar API - SC

2004-06-15 removed set_default_font from font_manager to unify font
           customization using the new function rc.  See API_CHANGES
           for more info.  The examples fonts_demo.py and
           fonts_demo_kw.py are ported to the new API - JDH

2004-06-15 Improved (yet again!) axis scaling to properly handle
           singleton plots - JDH

2004-06-15 Restored the old FigureCanvasGTK.draw() - SC

2004-06-11 More memory leak fixes in transforms and ft2font - JDH

2004-06-11 Eliminated numerix .numerix file and environment variable
       NUMERIX.  Fixed bug which prevented command line overrides:
       --numarray or --numeric. - JTM

2004-06-10 Added rc configuration function rc; deferred all rc param
           setting until object creation time; added new rc attrs:
           lines.markerfacecolor, lines.markeredgecolor,
           lines.markeredgewidth, patch.linewidth, patch.facecolor,
           patch.edgecolor, patch.antialiased; see
           examples/customize_rc.py for usage - JDH


---------------------------------------------------------------

2004-06-09 0.54.2 released

2004-06-08 Rewrote ft2font using CXX as part of general memory leak
           fixes; also fixed transform memory leaks  - JDH

2004-06-07 Fixed several problems with log ticks and scaling - JDH

2004-06-07 Fixed width/height issues for images - JDH

2004-06-03 Fixed draw_if_interactive bug for semilogx;

2004-06-02 Fixed text clipping to clip to axes - JDH

2004-06-02 Fixed leading newline text and multiple newline text - JDH

2004-06-02 Fixed plot_date to return lines - JDH

2004-06-01 Fixed plot to work with x or y having shape N,1 or 1,N - JDH

2004-05-31 Added renderer markeredgewidth attribute of Line2D. - ADS

2004-05-29 Fixed tick label clipping to work with navigation.

2004-05-28 Added renderer grouping commands to support groups in
           SVG/PS. - JDH

2004-05-28 Fixed, this time I really mean it, the singleton plot
           plot([0]) scaling bug; Fixed Flavio's shape = N,1 bug - JDH

2004-05-28 added colorbar - JDH

2004-05-28 Made some changes to the matplotlib.colors.Colormap to
           propertly support clim - JDH

-----------------------------------------------------------------

2004-05-27 0.54.1 released

2004-05-27 Lots of small bug fixes: rotated text at negative angles,
           errorbar capsize and autoscaling, right tick label
           position, gtkagg on win98, alpha of figure background,
           singleton plots - JDH

2004-05-26 Added Gary's errorbar stuff and made some fixes for length
           one plots and constant data plots - JDH

2004-05-25 Tweaked TkAgg backend so that canvas.draw() works
       more like the other backends.  Fixed a bug resulting
       in 2 draws per figure mangager show().      - JTM

------------------------------------------------------------

2004-05-19 0.54 released

2004-05-18 Added newline separated text with rotations to text.Text
           layout - JDH

2004-05-16 Added fast pcolor using PolyCollections.  - JDH

2004-05-14 Added fast polygon collections - changed scatter to use
           them.  Added multiple symbols to scatter.  10x speedup on
           large scatters using \*Agg and 5X speedup for ps. - JDH

2004-05-14 On second thought... created an "nx" namespace in
           in numerix which maps type names onto typecodes
           the same way for both numarray and Numeric.  This
           undoes my previous change immediately below. To get a
           typename for Int16 useable in a Numeric extension:
           say nx.Int16. - JTM

2004-05-15 Rewrote transformation class in extension code, simplified
           all the artist constructors - JDH

2004-05-14 Modified the type definitions in the numarray side of
       numerix so that they are Numeric typecodes and can be
       used with Numeric compilex extensions.  The original
       numarray types were renamed to type<old_name>.    - JTM

2004-05-06 Gary Ruben sent me a bevy of new plot symbols and markers.
           See matplotlib.matlab.plot - JDH

2004-05-06 Total rewrite of mathtext - factored ft2font stuff out of
           layout engine and defined abstract class for font handling
           to lay groundwork for ps mathtext.  Rewrote parser and made
           layout engine much more precise.  Fixed all the layout
           hacks.  Added spacing commands \/ and \hspace.  Added
           composite chars and defined angstrom. - JDH

2004-05-05 Refactored text instances out of backend; aligned
           text with arbitrary rotations is now supported - JDH

2004-05-05 Added a Matrix capability for numarray to numerix.  JTM

2004-05-04 Updated whats_new.html.template to use dictionary and
           template loop, added anchors for all versions and items;
           updated goals.txt to use those for links. PG

2004-05-04 Added fonts_demo.py to backend_driver, and AFM and TTF font
           caches to font_manager.py - PEB

2004-05-03 Redid goals.html.template to use a goals.txt file that
           has a pseudo restructured text organization. PG

2004-05-03 Removed the close buttons on all GUIs and added the python
           #! bang line to the examples following Steve Chaplin's
           advice on matplotlib dev

2004-04-29 Added CXX and rewrote backend_agg using it; tracked down
           and fixed agg memory leak - JDH

2004-04-29 Added stem plot command - JDH

2004-04-28 Fixed PS scaling and centering bug - JDH

2004-04-26 Fixed errorbar autoscale problem - JDH

2004-04-22 Fixed copy tick attribute bug, fixed singular datalim
           ticker bug; fixed mathtext fontsize interactive bug. - JDH

2004-04-21 Added calls to draw_if_interactive to axes(), legend(),
           and pcolor().  Deleted duplicate pcolor(). - JTM

------------------------------------------------------------

2004-04-21 matplotlib 0.53 release

2004-04-19 Fixed vertical alignment bug in PS backend - JDH

2004-04-17 Added support for two scales on the "same axes" with tick
           different ticking and labeling left right or top bottom.
           See examples/two_scales.py - JDH

2004-04-17 Added default dirs as list rather than single dir in
       setupext.py - JDH

2004-04-16 Fixed wx exception swallowing bug (and there was much
           rejoicing!) - JDH

2004-04-16 Added new ticker locator a formatter, fixed default font
           return - JDH

2004-04-16 Added get_name method to FontProperties class. Fixed font lookup
       in GTK and WX backends. - PEB

2004-04-16 Added get- and set_fontstyle msethods. - PEB

2004-04-10 Mathtext fixes: scaling with dpi,  - JDH

2004-04-09 Improved font detection algorithm. - PEB

2004-04-09 Move deprecation warnings from text.py to __init__.py - PEB

2004-04-09 Added default font customization - JDH

2004-04-08 Fixed viewlim set problem on axes and axis. - JDH

2004-04-07 Added validate_comma_sep_str and font properties paramaters to
           __init__.  Removed font families and added rcParams to
           FontProperties __init__ arguments in font_manager.  Added
           default font property parameters to .matplotlibrc file with
           descriptions.  Added deprecation warnings to the get\_ - and
           set_fontXXX methods of the Text object. - PEB

2004-04-06 Added load and save commands for ASCII data - JDH

2004-04-05 Improved font caching by not reading AFM fonts until needed.
           Added better documentation.  Changed the behaviour of the
           get_family, set_family, and set_name methods of FontProperties.
           - PEB

2004-04-05 Added WXAgg backend - JDH

2004-04-04 Improved font caching in backend_agg with changes to
           font_manager - JDH

2004-03-29 Fixed fontdicts and kwargs to work with new font manager -
           JDH

--------------------------------------------

This is the Old, stale, never used changelog

2002-12-10 - Added a TODO file and CHANGELOG.  Lots to do -- get
             crackin'!

           - Fixed y zoom tool bug

           - Adopted a compromise fix for the y data clipping problem.
             The problem was that for solid lines, the y data clipping
             (as opposed to the gc clipping) caused artifactual
             horizontal solid lines near the ylim boundaries.  I did a
             5% offset hack in Axes set_ylim functions which helped,
             but didn't cure the problem for very high gain y zooms.
             So I disabled y data clipping for connected lines .  If
             you need extensive y clipping, either plot(y,x) because x
             data clipping is always enabled, or change the _set_clip
             code to 'if 1' as indicated in the lines.py src.  See
             _set_clip in lines.py and set_ylim in figure.py for more
             information.


2002-12-11 - Added a measurement dialog to the figure window to
             measure axes position and the delta x delta y with a left
             mouse drag.  These defaults can be overridden by deriving
             from Figure and overrriding button_press_event,
             button_release_event, and motion_notify_event,
             and _dialog_measure_tool.

           - fixed the navigation dialog so you can check the axes the
             navigation buttons apply to.



2003-04-23 Released matplotlib v0.1

2003-04-24 Added a new line style PixelLine2D which is the plots the
           markers as pixels (as small as possible) with format
           symbol ','

           Added a new class Patch with derived classes Rectangle,
           RegularPolygon and Circle

2003-04-25 Implemented new functions errorbar, scatter and hist

           Added a new line type '|' which is a vline.  syntax is
           plot(x, Y, '|') where y.shape = len(x),2 and each row gives
           the ymin,ymax for the respective values of x.  Previously I
           had implemented vlines as a list of lines, but I needed the
           efficientcy of the numeric clipping for large numbers of
           vlines outside the viewport, so I wrote a dedicated class
           Vline2D which derives from Line2D


2003-05-01

   Fixed ytick bug where grid and tick show outside axis viewport with gc clip

2003-05-14

   Added new ways to specify colors 1) matlab format string 2)
   html-style hex string, 3) rgb tuple.  See examples/color_demo.py

2003-05-28

    Changed figure rendering to draw form a pixmap to reduce flicker.
    See examples/system_monitor.py for an example where the plot is
    continusouly updated w/o flicker.  This example is meant to
    simulate a system monitor that shows free CPU, RAM, etc...

2003-08-04

    Added Jon Anderson's GTK shell, which doesn't require pygtk to
    have threading built-in and looks nice!

2003-08-25

   Fixed deprecation warnings for python2.3 and pygtk-1.99.18

2003-08-26

   Added figure text with new example examples/figtext.py


2003-08-27

   Fixed bugs i figure text with font override dictionairies and fig
   text that was placed outside the window bounding box

2003-09-1 thru 2003-09-15

   Added a postscript and a GD module backend

2003-09-16

   Fixed font scaling and point scaling so circles, squares, etc on
   lines will scale with DPI as will fonts.  Font scaling is not fully
   implemented on the gtk backend because I have not figured out how
   to scale fonts to arbitrary sizes with GTK

2003-09-17

   Fixed figure text bug which crashed X windows on long figure text
   extending beyond display area.  This was, I believe, due to the
   vestigial erase functionality that was no longer needed since I
   began rendering to a pixmap

2003-09-30  Added legend

2003-10-01 Fixed bug when colors are specified with rgb tuple or hex
   string.


2003-10-21  Andrew Straw provided some legend code which I modified
      and incorporated.  Thanks Andrew!

2003-10-27 Fixed a bug in axis.get_view_distance that affected zoom in
  versus out with interactive scrolling, and a bug in the axis text
  reset system that prevented the text from being redrawn on a
  interactive gtk view lim set with the widget

  Fixed a bug in that prevented the manual setting of ticklabel
  strings from working properly

2003-11-02 - Do a nearest neighbor color pick on GD when
             allocate fails

2003-11-02
   - Added pcolor plot
   - Added MRI example
   - Fixed bug that screwed up label position if xticks or yticks were
     empty
   - added nearest neighbor color picker when GD max colors exceeded
   - fixed figure background color bug in GD backend

2003-11-10 - 2003-11-11
   - major refactoring.

     * Ticks (with labels, lines and grid) handled by dedicated class
     * Artist now know bounding box and dpi
     * Bounding boxes and transforms handled by dedicated classes
     * legend in dedicated class.  Does a better job of alignment and
       bordering.  Can be initialized with specific line instances.
       See examples/legend_demo2.py


2003-11-14 Fixed legend positioning bug and added new position args

2003-11-16 Finsihed porting GD to new axes API


2003-11-20 - add TM for matlab on website and in docs


2003-11-20 - make a nice errorbar and scatter screenshot

2003-11-20 - auto line style cycling for multiple line types
   broken

2003-11-18 (using inkrect) :logical rect too big on gtk backend

2003-11-18 ticks don't reach edge of axes in gtk mode --
   rounding error?

2003-11-20 - port Gary's errorbar code to new API before 0.40

2003-11-20 - problem with stale _set_font.  legend axes box
   doesn't resize on save in GTK backend -- see htdocs legend_demo.py

2003-11-21 - make a dash-dot dict for the GC

2003-12-15 - fix install path bug
