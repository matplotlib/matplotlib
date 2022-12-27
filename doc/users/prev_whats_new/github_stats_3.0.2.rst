.. _github-stats-3-0-2:

GitHub statistics for 3.0.2 (Nov 10, 2018)
==========================================

GitHub statistics for 2018/09/18 (tag: v3.0.0) - 2018/11/10

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed 170 issues and merged 224 pull requests.
The full list can be seen `on GitHub <https://github.com/matplotlib/matplotlib/milestone/39?closed=1>`__

The following 49 authors contributed 460 commits.

* Abhinuv Nitin Pitale
* Alon Hershenhorn
* Andras Deak
* Ankur Dedania
* Antony Lee
* Anubhav Shrimal
* Ayappan P
* azure-pipelines[bot]
* Ben Root
* Colin
* Colin Carroll
* Daniele Nicolodi
* David Haberthür
* David Stansby
* Dmitry Mottl
* Elan Ernest
* Elliott Sales de Andrade
* Eric Wieser
* esvhd
* Galen Lynch
* hannah
* Ildar Akhmetgaleev
* ImportanceOfBeingErnest
* Jody Klymak
* Joel Wanner
* Kai Muehlbauer
* Kevin Rose
* Kyle Sunden
* Marcel Martin
* Matthias Bussonnier
* MeeseeksMachine
* Michael Jancsy
* Nelle Varoquaux
* Nick Papior
* Nikita Kniazev
* Paul Hobson
* pharshalp
* Rasmus Diederichsen
* Ryan May
* saksmito
* Takafumi Arakaki
* teresy
* Thomas A Caswell
* thoo
* Tim Hoffmann
* Tobias Megies
* Tyler Makaro
* Will Handley
* Yuxin Wu

GitHub issues and pull requests:

Pull Requests (224):

* :ghpull:`12785`: Use level kwargs in irregular contour example
* :ghpull:`12767`: Make colorbars constructible with dataless ScalarMappables.
* :ghpull:`12775`: Add note to errorbar function about sign of errors
* :ghpull:`12776`: Fix typo in example (on-borad -> on-board).
* :ghpull:`12771`: Do not rely on external stack frame to exist
* :ghpull:`12526`: Rename jquery files
* :ghpull:`12552`: Update docs for writing image comparison tests.
* :ghpull:`12746`: Use skipif, not xfail, for uncomparable image formats.
* :ghpull:`12747`: Prefer log.warning("%s", ...) to log.warning("%s" % ...).
* :ghpull:`11753`: FIX: Apply aspect before drawing starts
* :ghpull:`12749`: Move toolmanager warning from logging to warning.
* :ghpull:`12708`: Run flake8 in a separate travis environment
* :ghpull:`12737`: Improve docstring of Arc
* :ghpull:`12598`: Support Cn colors with n>=10.
* :ghpull:`12670`: FIX: add setter for hold to un-break basemap
* :ghpull:`12693`: Workaround Text3D breaking tight_layout()
* :ghpull:`12727`: Reorder API docs: separate file per module
* :ghpull:`12738`: Add unobtrusive depreaction note to the first line of the docstring.
* :ghpull:`12740`: DOC: constrained layout guide (fix: Spacing with colorbars)
* :ghpull:`11663`: Refactor color parsing of Axes.scatter
* :ghpull:`12736`: Move deprecation note to end of docstring
* :ghpull:`12704`: Rename tkinter import from Tk to tk.
* :ghpull:`12730`: MNT: merge ignore lines in .flake8
* :ghpull:`12707`: Fix tk error when closing first pyplot figure
* :ghpull:`12715`: Cleanup dviread.
* :ghpull:`12717`: Delete some ``if __name__ == "__main__"`` clauses.
* :ghpull:`12726`: Fix test_non_gui_warning for Azure (and mplcairo).
* :ghpull:`12720`: Improve docs on Axes scales
* :ghpull:`12537`: Improve error message on failing test_pyplot_up_to_date
* :ghpull:`12721`: Make get_scale_docs() internal
* :ghpull:`12617`: Set up CI with Azure Pipelines
* :ghpull:`12673`: Fix for _axes.scatter() array index out of bound error
* :ghpull:`12676`: Doc: document textpath module
* :ghpull:`12705`: Improve docs on Axes limits and direction
* :ghpull:`12706`: Extend sphinx Makefile to cleanup completely
* :ghpull:`12481`: Warn if plot_surface Z values contain NaN
* :ghpull:`12709`: Correctly remove nans when drawing paths with pycairo.
* :ghpull:`12685`: Make ticks in demo_axes_rgb.py visible
* :ghpull:`12691`: DOC: Link to "How to make a PR" tutorials as badge and in contributing
* :ghpull:`12684`: Change ipython block to code-block
* :ghpull:`11974`: Make code match comment in sankey.
* :ghpull:`12440`: Make arguments to @deprecated/warn_deprecated keyword-only.
* :ghpull:`12683`: TST: mark test_constrainedlayout.py::test_colorbar_location as flaky
* :ghpull:`12686`: Remove deprecation warnings in tests
* :ghpull:`12470`: Update AutoDateFormatter with locator
* :ghpull:`12656`: FIX: fix error in colorbar.get_ticks not having valid data
* :ghpull:`12586`: Improve linestyles example
* :ghpull:`12006`: Added stacklevel=2 to all warnings.warn calls (issue 10643)
* :ghpull:`12651`: FIX: ignore non-finite bbox
* :ghpull:`12653`: Don't warn when accessing deprecated properties from the class.
* :ghpull:`12608`: ENH: allow matplotlib.use after getbackend
* :ghpull:`12658`: Do not warn-depreacted when iterating over rcParams
* :ghpull:`12635`: FIX: allow non bbox_extra_artists calls
* :ghpull:`12659`: Add note that developer discussions are private
* :ghpull:`12543`: Make rcsetup.py flak8 compliant
* :ghpull:`12642`: Don't silence TypeErrors in fmt_{x,y}data.
* :ghpull:`11667`: DOC: update doc requirement
* :ghpull:`12442`: Deprecate passing drawstyle with linestyle as single string.
* :ghpull:`12625`: Shorten some docstrings.
* :ghpull:`12627`: Be a bit more stringent on invalid inputs.
* :ghpull:`12561`: Properly css-style exceptions in the documentation
* :ghpull:`12629`: Fix issue with PyPy on macOS
* :ghpull:`10933`: Remove "experimental" fontconfig font_manager backend.
* :ghpull:`12630`: Fix RcParams.__len__
* :ghpull:`12285`: FIX: Don't apply tight_layout if axes collapse
* :ghpull:`12548`: undef _XOPEN_SOURCE breaks the build in AIX
* :ghpull:`12615`: Fix travis OSX build
* :ghpull:`12600`: Minor style fixes.
* :ghpull:`12607`: STY: fix whitespace and escaping
* :ghpull:`12603`: FIX: don't import macosx to check if eventloop running
* :ghpull:`12599`: Fix formatting of docstring
* :ghpull:`12569`: Don't confuse uintptr_t and Py_ssize_t.
* :ghpull:`12572`: Fix singleton hist labels
* :ghpull:`12581`: Fix hist() error message
* :ghpull:`12570`: Fix mathtext tutorial for build with Sphinx 1.8.
* :ghpull:`12487`: Update docs/tests for the deprecation of aname and label1On/label2On/etc.
* :ghpull:`12521`: Improve docstring of draw_idle()
* :ghpull:`12573`: BUG: mplot3d: Don't crash if azim or elev are non-integral
* :ghpull:`12574`: Remove some unused imports
* :ghpull:`12568`: Add note regarding builds of old Matplotlibs.
* :ghpull:`12555`: Clarify horizontalalignment and verticalalignment in suptitle
* :ghpull:`12547`: Disable sticky edge accumulation if no autoscaling.
* :ghpull:`12546`: Avoid quadratic behavior when accumulating stickies.
* :ghpull:`12159`: FIX: colorbar re-check norm before draw for autolabels
* :ghpull:`12501`: Rectified plot error
* :ghpull:`11789`: endless looping GIFs with PillowWriter
* :ghpull:`12525`: Fix some flake8 issues
* :ghpull:`12431`: FIX: allow single-string color for scatter
* :ghpull:`12216`: Doc: Fix search for sphinx >=1.8
* :ghpull:`12461`: FIX: make add_lines work with new colorbar
* :ghpull:`12241`: FIX: make unused spines invisible
* :ghpull:`12516`: Don't handle impossible values for ``align`` in hist()
* :ghpull:`12504`: DOC: clarify min supported version wording
* :ghpull:`12507`: FIX: make minor ticks formatted with science formatter as well
* :ghpull:`12500`: Adjust the widths of the messages during the build.
* :ghpull:`12492`: Simplify radar_chart example.
* :ghpull:`12478`: MAINT: NumPy deprecates asscalar in 1.16
* :ghpull:`12363`: FIX: errors in get_position changes
* :ghpull:`12495`: Fix duplicate condition in pathpatch3d example
* :ghpull:`11984`: Strip out pkg-config machinery for agg and libqhull.
* :ghpull:`12463`: Document Artist.cursor_data() parameter
* :ghpull:`12489`: Fix typo in documentation of ylim
* :ghpull:`12482`: Test slider orientation
* :ghpull:`12317`: Always install mpl_toolkits.
* :ghpull:`12246`: Be less tolerant of broken installs.
* :ghpull:`12477`: Use \N{MICRO SIGN} instead of \N{GREEK SMALL LETTER MU} in EngFormatter.
* :ghpull:`12483`: Kill FontManager.update_fonts.
* :ghpull:`12448`: Don't error if some font directories are not readable.
* :ghpull:`12474`: Throw ValueError when irregularly gridded data is passed to streamplot.
* :ghpull:`12469`: Clarify documentation of offsetbox.AnchoredText's prop kw argument
* :ghpull:`12468`: Fix ``set_ylim`` unit handling
* :ghpull:`12466`: np.fromstring -> np.frombuffer.
* :ghpull:`12369`: Improved exception handling on animation failure
* :ghpull:`12460`: Deprecate RendererBase.strip_math.
* :ghpull:`12457`: Fix tutorial typos.
* :ghpull:`12453`: Rollback erroneous commit to whats_new.rst from #10746
* :ghpull:`12452`: Minor updates to the FAQ.
* :ghpull:`10746`: Adjusted matplotlib.widgets.Slider to have optional vertical orientatation
* :ghpull:`12441`: Get rid of a signed-compare warning.
* :ghpull:`12430`: Deprecate Axes3D.plot_surface(shade=None)
* :ghpull:`12435`: Fix numpydoc parameter formatting
* :ghpull:`12434`: Clarify documentation for textprops keyword parameter of TextArea
* :ghpull:`12427`: Document Artist.get_cursor_data
* :ghpull:`12277`: FIX: datetime64 now recognized if in a list
* :ghpull:`10322`: Use np.hypot wherever possible.
* :ghpull:`12423`: Minor simplifications to backend_svg.
* :ghpull:`12293`: Make pyplot more tolerant wrt. 3rd-party subclasses.
* :ghpull:`12360`: Replace axes_grid by axes_grid1 in test
* :ghpull:`10356`: fix detecting which artist(s) the mouse is over
* :ghpull:`12416`: Move font cache rebuild out of exception handler
* :ghpull:`11891`: Group some print()s in backend_ps.
* :ghpull:`12165`: Remove deprecated mlab code
* :ghpull:`12394`: DOC: fix CL tutorial to give same output from saved file and example
* :ghpull:`12387`: Update HTML animation as slider is dragged
* :ghpull:`12408`: Don't crash on invalid registry font entries on Windows.
* :ghpull:`10088`: Deprecate Tick.{gridOn,tick1On,label1On,...} in favor of set_visible.
* :ghpull:`12149`: Mathtext tutorial fixes
* :ghpull:`12393`: Deprecate to-days converters in matplotlib dates
* :ghpull:`12257`: Document standard backends in matplotlib.use()
* :ghpull:`12383`: Revert change of parameter name in annotate()
* :ghpull:`12385`: CI: Added Appveyor Python 3.7 build
* :ghpull:`12247`: Machinery for deprecating properties.
* :ghpull:`12371`: Move check for ImageMagick Windows path to bin_path().
* :ghpull:`12384`: Cleanup axislines style.
* :ghpull:`12353`: Doc: clarify default parameters in scatter docs
* :ghpull:`12366`: TST: Update test images for new Ghostscript.
* :ghpull:`11648`: FIX: colorbar placement in constrained layout
* :ghpull:`12368`: Don't use stdlib private API in animation.py.
* :ghpull:`12351`: dviread: find_tex_file: Ensure the encoding on windows
* :ghpull:`12244`: Merge barchart examples.
* :ghpull:`12372`: Remove two examples.
* :ghpull:`12214`: Improve docstring of Annotation
* :ghpull:`12347`: DOC: add_child_axes to axes_api.rst
* :ghpull:`12304`: TST: Merge Qt tests into one file.
* :ghpull:`12321`: maint: setupext.py for freetype had a Catch case for missing ft2build.h
* :ghpull:`12340`: Catch test deprecation warnings for mlab.demean
* :ghpull:`12334`: Improve selection of inset indicator connectors.
* :ghpull:`12316`: Fix some warnings from Travis
* :ghpull:`12268`: FIX: remove unnecessary ``self`` in ``super_``-calls, fixes #12265
* :ghpull:`12212`: font_manager: Fixed problems with Path(...).suffix
* :ghpull:`12326`: fixed minor spelling error in docstring
* :ghpull:`12296`: Make FooConverter inherit from ConversionInterface in examples
* :ghpull:`12322`: Fix the docs build.
* :ghpull:`12319`: Fix Travis 3.6 builds
* :ghpull:`12309`: Deduplicate implementations of FooNorm.autoscale{,_None}
* :ghpull:`12314`: Deprecate ``axis('normal')`` in favor of ``axis('auto')``.
* :ghpull:`12313`: BUG: Fix typo in view_limits() for MultipleLocator
* :ghpull:`12307`: Clarify missing-property error message.
* :ghpull:`12274`: MNT: put back ``_hold`` as read-only attribute on AxesBase
* :ghpull:`12260`: Fix docs : change from issue #12191, remove "if 1:" blocks in examples 
* :ghpull:`12163`: TST: Defer loading Qt framework until test is run.
* :ghpull:`12253`: Handle utf-8 output by kpathsea on Windows.
* :ghpull:`12301`: Ghostscript 9.0 requirement revisited
* :ghpull:`12294`: Fix expand_dims warnings in triinterpolate
* :ghpull:`12292`: TST: Modify the bar3d test to show three more angles
* :ghpull:`12297`: Remove some pytest parameterising warnings
* :ghpull:`12261`: FIX:  parasite axis2 demo
* :ghpull:`12278`: Document inheriting docstrings
* :ghpull:`12262`: Simplify empty-rasterized pdf test.
* :ghpull:`12269`: Add some param docs to BlockingInput methods
* :ghpull:`12272`: Fix ``contrained`` to ``constrained``
* :ghpull:`12255`: Deduplicate inherited docstrings.
* :ghpull:`12254`: Improve docstrings of Animations
* :ghpull:`12258`: Fix CSS for module-level data
* :ghpull:`12222`: Remove extraneous if 1 statements in demo_axisline_style.py
* :ghpull:`12137`:  MAINT: Vectorize bar3d 
* :ghpull:`12219`: Merge OSXInstalledFonts into findSystemFonts.
* :ghpull:`12229`: Less ACCEPTS, more numpydoc.
* :ghpull:`12209`: Doc: Sort named colors example by palette
* :ghpull:`12237`: Use (float, float) as parameter type for 2D positions in docstrings
* :ghpull:`12238`: Typo in docs
* :ghpull:`12236`: Make boilerplate-generated pyplot.py flake8 compliant
* :ghpull:`12231`: CI: Speed up Appveyor repository cloning
* :ghpull:`12228`: Fix trivial typo in docs.
* :ghpull:`12227`: Use (float, float) as parameter type for 2D positions
* :ghpull:`12199`: Allow disabling specific mouse actions in blocking_input
* :ghpull:`12213`: Change win32InstalledFonts return value
* :ghpull:`12207`: FIX: dont' check for interactive framework if none required
* :ghpull:`11688`: Don't draw axis (spines, ticks, labels) twice when using parasite axes.
* :ghpull:`12210`: Axes.tick_params() argument checking
* :ghpull:`12211`: Fix typo
* :ghpull:`12200`: Slightly clarify some invalid shape exceptions for image data.
* :ghpull:`12151`: Don't pretend @deprecated applies to classmethods.
* :ghpull:`12190`: Remove some unused variables and imports
* :ghpull:`12186`: DOC: fix API note about get_tightbbox
* :ghpull:`12203`: Document legend's slowness when "best" location is used
* :ghpull:`12192`: Exclude examples from lgtm analysis
* :ghpull:`12196`: Give Carreau the ability to mention the backport bot.
* :ghpull:`12187`: DOC: Update INSTALL.rst
* :ghpull:`12164`: Fix Annotation.contains.
* :ghpull:`12177`: FIX: remove cwd from mac font path search
* :ghpull:`12182`: Fix Flash of Unstyled Content by removing remaining Flipcause integration
* :ghpull:`12184`: DOC: update "Previous What's New" for 2.2 with reference to cividis paper
* :ghpull:`12183`: Doc: Don't use Sphinx 1.8
* :ghpull:`12171`: Remove internal warning due to zsort deprecation
* :ghpull:`12166`: Document preference order for backend auto selection
* :ghpull:`12154`: Avoid triggering deprecation warnings with pytest 3.8.
* :ghpull:`12030`: Speed up canvas redraw for GTK3Agg backend.
* :ghpull:`12157`: Properly declare the interactive framework for the qt4foo backends.
* :ghpull:`12156`: Cleanup the GridSpec demos.
* :ghpull:`12144`: Add explicit getters and setters for Annotation.anncoords.
* :ghpull:`12152`: Use _warn_external for deprecations warnings.
* :ghpull:`12148`: BLD: pragmatic fix for building basic_unit example on py37
* :ghpull:`12147`: DOC: update the gh_stats code

Issues (170):

* :ghissue:`12699`: Annotations get cropped out of figures saved with bbox_inches='tight'
* :ghissue:`9217`: Weirdness with inline figure DPI settings in Jupyter Notebook
* :ghissue:`4853`: %matplotlib notebook creates much bigger figures than %matplotlib inline
* :ghissue:`12780`: Vague/misleading exception message in scatter()
* :ghissue:`10239`: Weird interaction with Tkinter
* :ghissue:`10045`: subplots_adjust() breaks layout of tick labels
* :ghissue:`12765`: Matplotlib draws incorrect color
* :ghissue:`11800`: Gridspec tutorial
* :ghissue:`12757`: up the figure
* :ghissue:`12724`: Importing pyplot steals focus on macOS 
* :ghissue:`12669`: fixing _hold on cartopy broke basemap
* :ghissue:`12687`: Plotting text on 3d axes before tight_layout() breaks tight_layout()
* :ghissue:`12734`: Wishlist: functionally linked twin axes
* :ghissue:`12576`: RcParams is fundamentally broken
* :ghissue:`12641`: ``_axes.py.scatter()`` array index out of bound / calling from ``seaborn``
* :ghissue:`12703`: Error when closing first of several pyplot figures in TkAgg
* :ghissue:`12728`: Deprecation Warnings
* :ghissue:`4124`: Provide canonical examples of mpl in web frameworks
* :ghissue:`10574`: Default color after setting alptha to Patch in legened
* :ghissue:`12702`: couldn't find or load Qt platform plugin "windows" in "".
* :ghissue:`11139`: "make clean" doesn't remove all the build doc files
* :ghissue:`12701`: semilogy with NaN prevents display of Title (cairo backend)
* :ghissue:`12696`: Process finished with exit code -1 due to matplotlib configuration
* :ghissue:`12692`: matplotlib.plot.show always blocks the execution of python script
* :ghissue:`12433`: Travis error is MacOS image tolerance of 0.005 for ``test_constrained_layout.py::test_colorbar_location``
* :ghissue:`10017`: unicode_literals considered harmful
* :ghissue:`12682`: using AxesImage.set_clim() shrinks the colorbar
* :ghissue:`12620`: Overlapping 3D objects
* :ghissue:`12680`: matplotlib ui in thread still blocked
* :ghissue:`11908`: Improve linestyle documentation
* :ghissue:`12650`: Deprecation warnings when calling help(matplotlib)
* :ghissue:`10643`: Most warnings calls do not set the stacklevel
* :ghissue:`12671`: make_axes_locatable breaks with matplotlib 3.0
* :ghissue:`12664`: plt.scatter crashes because overwrites the colors to an empty list
* :ghissue:`12188`:  matplotlib 3 pyplot on MacOS bounces rocket icon in dock
* :ghissue:`12648`: Regression when calling annotate with nan values for the position
* :ghissue:`12362`: In 3.0.0 backend cannot be set if 'get_backend()' is run first
* :ghissue:`12649`: Over-verbose deprecation warning about examples.directory
* :ghissue:`12661`: In version 3.0.0 make_axes_locatable + colorbar does not produce expected result
* :ghissue:`12634`: axes_grid1 axes have no keyword argument 'bbox_extra_artists'
* :ghissue:`12654`: Broken 'Developer Discussions' link
* :ghissue:`12657`: With v3.0.0 mpl_toolkits.axes_grid1.make_axes_locatable().append_axes breaks in Jupyter
* :ghissue:`12645`: Markers are offset when 'facecolor' or 'edgecolor' are set to 'none' when plotting data
* :ghissue:`12644`: Memory leak with plt.plot in Jupyter Notebooks?
* :ghissue:`12632`: Do we need input hooks macosx?
* :ghissue:`12535`: AIX Support - Do not undef _XOPEN_SOURCE 
* :ghissue:`12626`: AttributeError: module 'matplotlib' has no attribute 'artist'
* :ghissue:`11034`: Doc Typo:  matplotlib.axes.Axes.get_yticklabels  / Axis.get_ticklabels
* :ghissue:`12624`: make_axes_locatable : Colorbar in the middle instead of bottom while saving a pdf, png.
* :ghissue:`11094`: can not use GUI backends inside django request handlers
* :ghissue:`12613`: transiently linked interactivity of unshared pair of axes generated with make_axes_locatable 
* :ghissue:`12578`: macOS builds are broken
* :ghissue:`12612`: gui backends do not work inside of flask request handlers
* :ghissue:`12611`: Matplotlib 3.0.0 Likely bug TypeError: stackplot() got multiple values for argument 'x'
* :ghissue:`12610`: matplotlibrc causes import to fail 3.0.0 (didn't crash 2.y.z series)
* :ghissue:`12601`: Can't import matplotlib
* :ghissue:`12597`: Please soon add Chinese language support!! It's to difficult for new people handle character
* :ghissue:`12590`: Matplotlib pypi distribution lacks packages for Python 2.7
* :ghissue:`3869`: Numeric labels do not work with plt.hist
* :ghissue:`12580`: Incorrect hist error message with bad color size
* :ghissue:`12100`: document where to get nightly wheels
* :ghissue:`7205`: Converting docstrings to numpydoc
* :ghissue:`12564`: Saving plot as PNG file prunes tick labels 
* :ghissue:`12161`: Problems of using sharex options with lines plots and colormesh with colorbar
* :ghissue:`12256`: tight_layout for plot with non-clipped screen-unit items causes issues on zoom
* :ghissue:`12545`: Program quit unormally without reporting error
* :ghissue:`12532`: Incorrect rendering of math symbols
* :ghissue:`12567`: Calling pyplot.show() with TkAgg backend on x86 machine raises OverflowError.
* :ghissue:`12571`: cannot install because Fatal Python error: initfsencoding: Unable to get the locale encoding
* :ghissue:`12566`: Problem installing Version 1.3.1 -> missing pkg-config freetype and libagg
* :ghissue:`12556`: Matplotlib 3.0.0 import hangs in clean environment
* :ghissue:`12197`: Weird behaviour of suptitle() when horizontalalignment is not 'center'
* :ghissue:`12550`: colorbar resizes in animation
* :ghissue:`12155`: Incorrect placement of Colorbar ticks using LogNorm
* :ghissue:`11787`: Looping gifs with PillowWriter
* :ghissue:`12533`: Plotting with alpha=0 with rasterized=True causes ValueError on saving to pdf
* :ghissue:`12438`: Scatter doesn't accept a list of strings as color spec.  
* :ghissue:`12429`: scatter() does not accept gray strings anymore
* :ghissue:`12499`: run my code failed after i Import pylab failed, python version is 3.6.6
* :ghissue:`12458`: add_lines misses lines for matplotlib.colorbar.ColorbarBase
* :ghissue:`12239`: 3d axes are collapsed by tight_layout
* :ghissue:`12414`: Function to draw angle between two lines
* :ghissue:`12488`: inconsistent colorbar tick labels for LogNorm
* :ghissue:`12515`: pyplot.step broken in 3.0.0?
* :ghissue:`12355`: Error for bbox_inches='tight' in savefig with make_axes_locatable
* :ghissue:`12505`: ImageGrid in 3.0
* :ghissue:`12502`: How can I put the ticks of logarithmic coordinate in the axes?
* :ghissue:`12496`: Maplotlib Can't Plot a Dataset
* :ghissue:`12486`: rotate label of legend ?
* :ghissue:`12291`: Importing pyplot crashes on macOS due to missing fontlist-v300.json and then Permission denied: '/opt/local/share/fonts'
* :ghissue:`12480`: "close_event" for nbagg/notebook backend
* :ghissue:`12467`: Documentation of AnchoredText's prop keyword argument is misleading
* :ghissue:`12288`: New function signatures in pyplot break Cartopy
* :ghissue:`12445`: Error on colorbar
* :ghissue:`8760`: Traceback from animation.MovieWriter.saving method is confusing because it provides no useful information
* :ghissue:`9205`: after the animation encoder (e.g. ffmpeg) fails, the animation framework itself fails internally in various ways while trying to report the error
* :ghissue:`12357`: Unclear error when saving Animation using FFMpeg
* :ghissue:`12454`: Formatting numerical legend
* :ghissue:`9636`: matplotlib crashes upon window resize
* :ghissue:`11473`: Continuous plotting cause memory leak 20-50kb/sec
* :ghissue:`12018`: No image pop-up or display for plt.imshow() and plt.show()
* :ghissue:`11583`: How to draw parallelepiped with real size scaling?
* :ghissue:`12446`: Polar Contour - float() argument must be a string or a number, not 'AxesParasiteParasiteAuxTrans'
* :ghissue:`12444`: Issues with gridspec/tight_layout in matplotlib version 2.2.3
* :ghissue:`11154`: Unexpected behavior for Axes3D.plot_surface(shade=None)
* :ghissue:`12409`: Calling savefig() multiple times causes crash of Spyder IDE / IPython Kernel dying.
* :ghissue:`9799`: FigureCanvasTkAgg - "buffer is of wrong type" error during blit
* :ghissue:`12439`: FileNotFoundError for font_manager
* :ghissue:`12437`: matplotlib-mac
* :ghissue:`12121`: Documentation of TextArea's fontprops keyword argument is misleading
* :ghissue:`12279`: Axes.format_cursor_data lacks documentation and seems unused
* :ghissue:`12428`: Simple plot spacing bug: ylabel gets wrongfully removed from plot
* :ghissue:`11190`: Images in the docs are too large.
* :ghissue:`12271`: error with errorbar with datetime64 
* :ghissue:`12405`: plt.stackplot() does not work with 3.0.0
* :ghissue:`12282`: ``Axes.imshow`` tooltip does not get updated when another call to ``Axes.imshow`` is made
* :ghissue:`12420`: How to remove Rectangle Selector from figure?
* :ghissue:`12391`: Constrained Layout tutorial needs some cleanup....
* :ghissue:`12406`: Bug with font finding, and here is my fix as well.
* :ghissue:`9051`: ParasiteAxes over plotting
* :ghissue:`12325`: Annotation change from "s" to "text" in 3.0- documentation
* :ghissue:`12397`: plt.show( ) not working (can't get figures to display in external window) when using jupyter QTconsole
* :ghissue:`12396`: Defining arrowprops in draggable annotation disables the pick_event
* :ghissue:`12389`: Setting row edge color of matplotlib table
* :ghissue:`12376`: The output figure file is strange: there is a lot of blank area on the output figure.
* :ghissue:`11641`: constrained_layout and colorbar for a subset of axes
* :ghissue:`12373`: Unexpected outcome with matplotlib.pyplot.pcolor()
* :ghissue:`12370`: ImageGrid bug when using inline backend
* :ghissue:`12364`: pdf image generated by matplotlib with semi transparent lines missing in Word on Windows.
* :ghissue:`12352`: TeX rendering broken on master with windows
* :ghissue:`12354`: Too many levels of symbolic links
* :ghissue:`12323`: indicate_inset_zoom sometimes draws incorrect connector lines
* :ghissue:`12341`: Figures not rendering in docker
* :ghissue:`12335`: Matplotlib plt.Rectangle Incoherent Results
* :ghissue:`12265`: ParasiteAxesAuxTrans  pcolor/pcolormesh and contour/contourf broken
* :ghissue:`12337`: AttributeError: module 'matplotlib.pyplot' has no attribute 'hold'
* :ghissue:`11673`: Inconsistent font settings when changing style context
* :ghissue:`11693`: The rcParams setting for figure.figsize does not change when run from another notebook
* :ghissue:`11725`: New mode between non-interactive and interactive?
* :ghissue:`12134`: tight_layout flips images when making plots without displaying them
* :ghissue:`12310`: plot fails with datetime64[ns] timezone aware objects (for example datetime64[ns, UTC+00:00] )
* :ghissue:`12191`: "if 1:" blocks in examples
* :ghissue:`11288`: FR: Figure.subplots add optional SubplotSpec parameter
* :ghissue:`12298`: c and cmap for plot
* :ghissue:`12286`: Sample code given in Matplotlib's site does not work.
* :ghissue:`11955`: UnicodeDecodeError on importing pyplot in python2
* :ghissue:`12208`: parasite axis2 demo now crashes with log x-axis
* :ghissue:`8871`: Error when using quantities when plotting errorbars
* :ghissue:`6658`: literature reference for 'viridis' colormap
* :ghissue:`6789`: Tutorial pyplot_scales.py crashes when used with plt.tight_layout()
* :ghissue:`6922`: imshow does not immediately update shared axes
* :ghissue:`11879`: Unable to change filename when saving from figure window
* :ghissue:`12225`: In histogram, bars whose count is larger than 2**31 sometimes become negative
* :ghissue:`1461`: DOC: keyword arguments to plt.axes, plt.subpot, and fig.add_subplot
* :ghissue:`12173`: Cannot import pyplot
* :ghissue:`12217`: Python will suddenly not plot anymore
* :ghissue:`12120`: Default legend behavior (loc='best') very slow for large amounts of data.
* :ghissue:`12176`: import pyplot on MacOS without font cache will search entire subtree of current dir
* :ghissue:`12146`: fix pdf docs
* :ghissue:`12160`: MacOS: Cannot import name 'format_exc'
* :ghissue:`12169`: Cannot install 3.0.0 "python setup.py egg_info" failed (freetype & png)
* :ghissue:`12168`: pip install v3.0.0 'failed with exit status 1181'
* :ghissue:`12107`: warnings re: deprecated pytest API with pytest 3.8
* :ghissue:`12162`: https://matplotlib.org/users/beginner.html is outdated
* :ghissue:`12010`: Popover over plot is very slow
* :ghissue:`6739`: Make matplotlib fail more gracefully in headless environments
* :ghissue:`3679`: Runtime detection for default backend
* :ghissue:`11340`: matplotlib fails to install from source with intel compiler
* :ghissue:`11838`: docs do not build on py3.7 due to small change in python handling of -m
* :ghissue:`12115`: Plot in JS Animation has larger margin than "normal" PNG plot
