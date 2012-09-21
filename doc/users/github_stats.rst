.. _github-stats:

Github stats
============

GitHub stats for 2012/06/30 - 2012/09/07 (tag: v1.1.1)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 71 authors contributed 1151 commits.

* Aaron Boushley
* Ahmet Bakan
* Amy
* Andrew Dawson
* Arnaud Gardelein
* Ben Gamari
* Ben Root
* Bradley M. Froehle
* Brett Graham
* Bussonnier Matthias
* C\. Gohlke
* Christoph Dann
* Christoph Gohlke
* Corey Farwell
* Craig M
* Craig Tenney
* Damon McDougall
* Daniel Hyams
* Darren Dale
* David Huard
* Eric Firing
* Ezra Peisach
* Gellule Xg
* Graham Poulter
* Hubert Holin
* Ian Thomas
* Ignas Anikevicius (gns_ank)
* Jack (aka Daniel) Kelly
* Jack Kelly
* Jae-Joon Lee
* James R. Evans
* Jason Grout
* Jens H. Nielsen
* Joe Kington
* John Hunter
* Jonathan Waltman
* Jouni K. Seppänen
* Lance Hepler
* Marc Abramowitz
* Martin Spacek
* Matthew Emmett
* Matthias BUSSONNIER
* Michael Droettboom
* Michiel de Hoon
* Mike Kaufman
* Neil
* Nelle Varoquaux
* Nikolay Vyahhi
* Paul Ivanov
* Peter Würtz
* Phil Elson
* Piti Ongmongkolkul
* Robert Johansson
* Russell Owen
* Ryan May
* Simon Cross
* Stefan van der Walt
* Takafumi Arakaki
* Thomas A Caswell
* Thomas Kluyver
* Thomas Robitaille
* Tobias Hoppe
* Tony S Yu
* Zach Pincus
* bev-a-tron
* endolith
* goir
* mcelrath
* pelson
* pwuertz
* vbr


We closed a total of 349 issues, 123 pull requests and 226 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (123):

* :ghpull:`1168`: PEP8 compliance on artist.py
* :ghpull:`1213`: Include username in tempdir
* :ghpull:`1182`: Bezier pep8
* :ghpull:`1206`: README and links fixes
* :ghpull:`1192`: Issue835 2: replacement for #835
* :ghpull:`1187`: Add a *simple* arrow example
* :ghpull:`1120`: FAIL: matplotlib.tests.test_transforms.test_pre_transform_plotting.test on Python 3.x
* :ghpull:`714`: Initial rework of gen_gallery.py
* :ghpull:`1150`: the affine matrix is calculated in the display coordinate for interpolation='none'
* :ghpull:`1145`: Fix formatter reset when twin{x,y}() is called
* :ghpull:`1201`: Fix typo in object-oriented API
* :ghpull:`1061`: Add log option to Axes.hist2d
* :ghpull:`1125`: Reduce object-oriented boilerplate for users
* :ghpull:`1195`: Fixed pickle tests to use the BufferIO object for python3 support.
* :ghpull:`1198`: Fixed python2.6 support (by removing use of viewvalues on a dict).
* :ghpull:`1197`: Handled future division changes for python3 (fixes #1194).
* :ghpull:`1162`: FIX nose.tools.assert_is is only supported with python2.7
* :ghpull:`803`: Return arrow collection as 2nd argument of streamplot.
* :ghpull:`1189`: BUG: Fix streamplot when velocity component is exactly zero.
* :ghpull:`1191`: Small bugfixes to the new pickle support.
* :ghpull:`1146`: Fix invalid transformation in InvertedSymmetricalLogTransform.
* :ghpull:`1169`: Subplot.twin[xy] returns a Subplot instance
* :ghpull:`1183`: FIX undefined elements were used at several places in the mlab module
* :ghpull:`498`: get_sample_data still broken on v.1.1.x
* :ghpull:`1170`: Uses tight_layout.get_subplotspec_list to check if all axes are compatible w/ tight_layout
* :ghpull:`1174`: closes #1173 - backporting python2.7 subprocess's check_output to be abl...
* :ghpull:`1175`: Pickling support added. Various whitespace fixes as a result of reading *lots* of code.
* :ghpull:`1098`: suppress exception upon quitting with qt4agg on osx
* :ghpull:`1171`: backend_pgf: handle OSError when testing for xelatex/pdflatex
* :ghpull:`1164`: doc: note contourf hatching in whats_new.rst
* :ghpull:`1153`: PEP8 on artist
* :ghpull:`1163`: tight_layout: fix regression for figures with non SubplotBase Axes
* :ghpull:`1159`: FIX assert_raises cannot be called with ``with``
* :ghpull:`1160`: backend_pgf: clarifications and fixes in documentation
* :ghpull:`1154`: six inclusion for dateutil on py3 doesn't work
* :ghpull:`1149`: Add Phil Elson's percentage histogram example
* :ghpull:`1158`: FIX - typo in lib/matplotlib/testing/compare.py
* :ghpull:`1155`: workaround for fixed dpi assumption in adjust_bbox_pdf
* :ghpull:`1142`: What's New: Python 3 paragraph
* :ghpull:`1130`: Fix writing pdf on stdout
* :ghpull:`832`: MPLCONFIGDIR tries to be created in read-only home
* :ghpull:`1140`: BUG: Fix fill_between when NaN values are present
* :ghpull:`1144`: Added tripcolor whats_new section.
* :ghpull:`1010`: Port part of errorfill from Tony Yu's mpltools.
* :ghpull:`1141`: backend_pgf: fix parentheses typo
* :ghpull:`1114`: Make grid accept alpha rcParam
* :ghpull:`1124`: PGF backend, fix #1116, #1118 and #1128
* :ghpull:`983`: Issues with dateutil and pytz
* :ghpull:`1133`: figure.py: import warnings, and make imports absolute
* :ghpull:`1132`: clean out obsolete matplotlibrc-related bits to close #1123
* :ghpull:`1131`: Cleanup after the gca test.
* :ghpull:`563`: sankey.add() has mutable defaults
* :ghpull:`731`: Plot limit with transform
* :ghpull:`1107`: Added %s support for labels.
* :ghpull:`774`: Allow automatic use of tight_layout.
* :ghpull:`1122`: DOC: Add streamplot description to What's New page
* :ghpull:`1111`: Fixed transoffset example from failing.
* :ghpull:`840`: Documentation Errors for specgram
* :ghpull:`1088`: For a text artist, if it has a _bbox_patch associated with it, the contains test should reflect this.
* :ghpull:`986`: Add texinfo build target in doc/make.py
* :ghpull:`1076`: PGF backend for XeLaTeX/LuaLaTeX support
* :ghpull:`1090`: External transform api
* :ghpull:`1108`: Fix documentation warnings
* :ghpull:`861`: Add rcfile function (which loads rc params from a given file).
* :ghpull:`1062`: increased the padding on FileMovieWritter.frame_format_str
* :ghpull:`1100`: Doc multi version master
* :ghpull:`1105`: Fixed comma between tests.
* :ghpull:`1095`: Colormap byteorder bug
* :ghpull:`1103`: colorbar: correct error introduced in commit 089024; closes #1102
* :ghpull:`1067`: Support multi-version documentation on the website
* :ghpull:`1031`: Added 'capthick' kwarg to errorbar()
* :ghpull:`1074`: Added broadcasting support in some mplot3d methods
* :ghpull:`1064`: Locator interface
* :ghpull:`850`: Added tripcolor triangle-centred colour values.
* :ghpull:`1093`: Exposed the callback id for the default key press handler so that it can be easily diabled. Fixes #215.
* :ghpull:`1065`: fixed conversion from pt to inch in tight_layout
* :ghpull:`1082`: doc: in pcolormesh docstring, say what it does.
* :ghpull:`1078`: doc: note that IDLE doesn't work with interactive mode.
* :ghpull:`1071`: patches.polygon: fix bug in handling of path closing, #1018.
* :ghpull:`1057`: Contour norm scaling
* :ghpull:`1056`: Test framework cleanups
* :ghpull:`778`: Make tests faster
* :ghpull:`1024`: broken links in the gallery
* :ghpull:`1054`:  stix_fonts_demo.py fails with bad refcount
* :ghpull:`960`: Fixed logformatting for non integer bases.
* :ghpull:`897`: GUI icon in Tkinter
* :ghpull:`1053`: Move Python 3 import of reload() to the module that uses it
* :ghpull:`1049`: Update examples/user_interfaces/embedding_in_wx2.py
* :ghpull:`1050`: Update examples/user_interfaces/embedding_in_wx4.py
* :ghpull:`1051`: Update examples/user_interfaces/mathtext_wx.py
* :ghpull:`1052`: Update examples/user_interfaces/wxcursor_demo.py
* :ghpull:`1047`: Enable building on Python 3.3 for Windows
* :ghpull:`1036`: Move all figures to the front with a non-interactive show() in macosx backend.
* :ghpull:`1042`: Three more plot_directive configuration options
* :ghpull:`1022`: contour: map extended ranges to "under" and "over" values
* :ghpull:`1007`: modifying GTK3 example to use pygobject, and adding a simple example to demonstrate NavigationToolbar in GTK3
* :ghpull:`1004`: Added savefig.bbox option to matplotlibrc
* :ghpull:`976`: Fix embedding_in_qt4_wtoolbar.py on Python 3
* :ghpull:`1034`: MdH = allow compilation on recent Mac OS X without compiler warnings
* :ghpull:`1028`: Fix use() so that it is possible to reset the rcParam.
* :ghpull:`1033`: Py3k: reload->imp.reload
* :ghpull:`1002`: Fixed potential overflow exception in the lines.contains() method
* :ghpull:`1025`: Timers
* :ghpull:`989`: Animation subprocess bug
* :ghpull:`898`: Added warnings for easily confusible subplot/subplots invokations
* :ghpull:`963`: Add detection of file extension for file-like objects
* :ghpull:`973`: Fix sankey.py pep8 and py3 compatibility
* :ghpull:`972`: Force closing PIL image files
* :ghpull:`981`: Fix pathpatch3d_demo.py on Python 3
* :ghpull:`980`: Fix basic_units.py on Python 3. PEP8 and PyLint cleanup.
* :ghpull:`1014`: qt4: remove duplicate file save button; and remove trailing whitespace
* :ghpull:`1011`: fix for bug #996 and related issues
* :ghpull:`985`: support current and future FreeBSD releases
* :ghpull:`1000`: Fix traceback for vlines/hlines, when an empty list or array passed in for x/y.
* :ghpull:`994`: Fix bug in pcolorfast introduced by #901
* :ghpull:`993`: Fix typo
* :ghpull:`908`: use Qt window title as default savefig filename
* :ghpull:`971`: Close fd temp file following rec2csv_bad_shape test
* :ghpull:`851`: Simple GUI interface enhancements
* :ghpull:`979`: Fix test_mouseclicks.py on Python 3
* :ghpull:`977`: Fix lasso_selector_demo.py on Python 3
* :ghpull:`970`: Fix tiff and jpeg export via PIL
* :ghpull:`961`: Issue 807 auto minor locator

Issues (226):

* :ghissue:`1096`: Documentation bug: pyplot.arrow does not list enough keywords to successfully draw an arrow
* :ghissue:`1168`: PEP8 compliance on artist.py
* :ghissue:`1213`: Include username in tempdir
* :ghissue:`1182`: Bezier pep8
* :ghissue:`1177`: Handled baseline image folder identification for non matplotlib projects.
* :ghissue:`1091`: Update README.txt for v1.2
* :ghissue:`1206`: README and links fixes
* :ghissue:`1192`: Issue835 2: replacement for #835
* :ghissue:`1187`: Add a *simple* arrow example
* :ghissue:`1120`: FAIL: matplotlib.tests.test_transforms.test_pre_transform_plotting.test on Python 3.x
* :ghissue:`835`: add documentation for figure show method in backend_bases and backend_template
* :ghissue:`714`: Initial rework of gen_gallery.py
* :ghissue:`1150`: the affine matrix is calculated in the display coordinate for interpolation='none'
* :ghissue:`1087`: Update whats new section
* :ghissue:`385`: BUG: plot_directive: look for plot script files relative to the .rst file
* :ghissue:`1110`: twiny overrides formatter and adds another x-axis
* :ghissue:`1145`: Fix formatter reset when twin{x,y}() is called
* :ghissue:`547`: undocumented scatter marker definition change
* :ghissue:`1201`: Fix typo in object-oriented API
* :ghissue:`1061`: Add log option to Axes.hist2d
* :ghissue:`1094`: Feature request - make it simpler to use full OO interface
* :ghissue:`1125`: Reduce object-oriented boilerplate for users
* :ghissue:`1085`: Images shifted relative to other plot feature in vector graphic output formats
* :ghissue:`1195`: Fixed pickle tests to use the BufferIO object for python3 support.
* :ghissue:`1198`: Fixed python2.6 support (by removing use of viewvalues on a dict).
* :ghissue:`1194`: Streamplot result python version dependent
* :ghissue:`1197`: Handled future division changes for python3 (fixes #1194).
* :ghissue:`557`: Crash during date axis setup
* :ghissue:`600`: errorbar(): kwarg 'markevery' not working as expected.
* :ghissue:`174`: Memory leak in example simple_idle_wx.py
* :ghissue:`232`: format in plot_direcitive sphinx>=1.0.6 compatible patch
* :ghissue:`1162`: FIX nose.tools.assert_is is only supported with python2.7
* :ghissue:`1165`: tight_layout fails on twinx, twiny
* :ghissue:`803`: Return arrow collection as 2nd argument of streamplot.
* :ghissue:`1189`: BUG: Fix streamplot when velocity component is exactly zero.
* :ghissue:`1191`: Small bugfixes to the new pickle support.
* :ghissue:`323`: native format for figures
* :ghissue:`1146`: Fix invalid transformation in InvertedSymmetricalLogTransform.
* :ghissue:`1169`: Subplot.twin[xy] returns a Subplot instance
* :ghissue:`1183`: FIX undefined elements were used at several places in the mlab module
* :ghissue:`498`: get_sample_data still broken on v.1.1.x
* :ghissue:`1170`: Uses tight_layout.get_subplotspec_list to check if all axes are compatible w/ tight_layout
* :ghissue:`1173`: The PGF backend only works on python2.7 and +
* :ghissue:`1174`: closes #1173 - backporting python2.7 subprocess's check_output to be abl...
* :ghissue:`1175`: Pickling support added. Various whitespace fixes as a result of reading *lots* of code.
* :ghissue:`1179`: Attempt at making travis output shorter.
* :ghissue:`1020`: Picklable figures
* :ghissue:`1098`: suppress exception upon quitting with qt4agg on osx
* :ghissue:`1171`: backend_pgf: handle OSError when testing for xelatex/pdflatex
* :ghissue:`1164`: doc: note contourf hatching in whats_new.rst
* :ghissue:`606`: Unable to configure grid using axisartist
* :ghissue:`1153`: PEP8 on artist
* :ghissue:`1163`: tight_layout: fix regression for figures with non SubplotBase Axes
* :ghissue:`1117`: ERROR: matplotlib.tests.test_axes.test_contour_colorbar.test fails on Python 3
* :ghissue:`1159`: FIX assert_raises cannot be called with ``with``
* :ghissue:`206`: hist normed=True problem?
* :ghissue:`1160`: backend_pgf: clarifications and fixes in documentation
* :ghissue:`1154`: six inclusion for dateutil on py3 doesn't work
* :ghissue:`320`: hist plot in percent
* :ghissue:`1149`: Add Phil Elson's percentage histogram example
* :ghissue:`1158`: FIX - typo in lib/matplotlib/testing/compare.py
* :ghissue:`1135`: Problems with bbox_inches='tight'
* :ghissue:`1155`: workaround for fixed dpi assumption in adjust_bbox_pdf
* :ghissue:`1142`: What's New: Python 3 paragraph
* :ghissue:`1138`: tight_bbox made assumptions about the display-units without checking the figure
* :ghissue:`1130`: Fix writing pdf on stdout
* :ghissue:`832`: MPLCONFIGDIR tries to be created in read-only home
* :ghissue:`1140`: BUG: Fix fill_between when NaN values are present
* :ghissue:`1144`: Added tripcolor whats_new section.
* :ghissue:`1010`: Port part of errorfill from Tony Yu's mpltools.
* :ghissue:`1141`: backend_pgf: fix parentheses typo
* :ghissue:`1114`: Make grid accept alpha rcParam
* :ghissue:`1118`: ERROR: matplotlib.tests.test_backend_pgf.test_pdflatex on Python 3.x
* :ghissue:`1116`: ERROR: matplotlib.tests.test_backend_pgf.test_xelatex
* :ghissue:`1124`: PGF backend, fix #1116, #1118 and #1128
* :ghissue:`745`: Cannot run tests with Python 3.x on MacOS 10.7
* :ghissue:`983`: Issues with dateutil and pytz
* :ghissue:`1137`: PGF/Tikz: savefig could not handle a filename
* :ghissue:`1128`: PGF back-end fails on simple graph
* :ghissue:`1133`: figure.py: import warnings, and make imports absolute
* :ghissue:`1123`: Rationalize the number of ancillary (default matplotlibrc) files
* :ghissue:`1132`: clean out obsolete matplotlibrc-related bits to close #1123
* :ghissue:`1131`: Cleanup after the gca test.
* :ghissue:`563`: sankey.add() has mutable defaults
* :ghissue:`238`: patch for qt4 backend
* :ghissue:`731`: Plot limit with transform
* :ghissue:`1107`: Added %s support for labels.
* :ghissue:`720`: Bug with bbox_inches='tight'
* :ghissue:`1084`: doc/mpl_examples/pylab_examples/transoffset.py not working as expected
* :ghissue:`774`: Allow automatic use of tight_layout.
* :ghissue:`1122`: DOC: Add streamplot description to What's New page
* :ghissue:`1111`: Fixed transoffset example from failing.
* :ghissue:`840`: Documentation Errors for specgram
* :ghissue:`1088`: For a text artist, if it has a _bbox_patch associated with it, the contains test should reflect this.
* :ghissue:`1119`: ERROR: matplotlib.tests.test_image.test_imread_pil_uint16 on Python 3.x
* :ghissue:`353`: Improved output of text in SVG and PDF
* :ghissue:`291`: size information from print_figure
* :ghissue:`986`: Add texinfo build target in doc/make.py
* :ghissue:`1076`: PGF backend for XeLaTeX/LuaLaTeX support
* :ghissue:`1090`: External transform api
* :ghissue:`1108`: Fix documentation warnings
* :ghissue:`811`: Allow tripcolor to directly plot triangle-centered functions
* :ghissue:`1005`: imshow with big-endian data types on OS X
* :ghissue:`892`: Update animation.py docstrings to "raw" Sphinx format
* :ghissue:`861`: Add rcfile function (which loads rc params from a given file).
* :ghissue:`988`: Trim white spaces while saving from Navigation toolbar
* :ghissue:`670`: Add a printer button to the navigation toolbar
* :ghissue:`1062`: increased the padding on FileMovieWritter.frame_format_str
* :ghissue:`188`: MacOSX backend brings up GUI unnecessarily
* :ghissue:`1041`: make.osx SDK location needs updating
* :ghissue:`1043`: Fix show command for Qt backend to raise window to top
* :ghissue:`1046`: test failing on master
* :ghissue:`962`: Bug with figure.savefig(): using AGG, PIL, JPG and StringIO
* :ghissue:`1045`: 1.1.1 not in pypi
* :ghissue:`1100`: Doc multi version master
* :ghissue:`1106`: Published docs for v1.1.1 missing pyplot.polar
* :ghissue:`569`: 3D bar graphs with variable depth
* :ghissue:`359`: new plot style: stackplot
* :ghissue:`297`: pip/easy_install installs old version of matplotlib
* :ghissue:`152`: Scatter3D: arguments (c,s,...) are not taken into account
* :ghissue:`1105`: Fixed comma between tests.
* :ghissue:`1095`: Colormap byteorder bug
* :ghissue:`1102`: examples/pylab_examples/contour_demo.py fails
* :ghissue:`1103`: colorbar: correct error introduced in commit 089024; closes #1102
* :ghissue:`1067`: Support multi-version documentation on the website
* :ghissue:`1031`: Added 'capthick' kwarg to errorbar()
* :ghissue:`1074`: Added broadcasting support in some mplot3d methods
* :ghissue:`1032`: Axesbase
* :ghissue:`1064`: Locator interface
* :ghissue:`850`: Added tripcolor triangle-centred colour values.
* :ghissue:`1059`: Matplotlib figure window freezes during interactive mode
* :ghissue:`215`: skipping mpl-axes-interaction during  key_press_event\'s
* :ghissue:`1093`: Exposed the callback id for the default key press handler so that it can be easily diabled. Fixes #215.
* :ghissue:`909`: Log Formatter for tick labels can't handle non-integer base
* :ghissue:`1065`: fixed conversion from pt to inch in tight_layout
* :ghissue:`1086`: Problem with subplot / matplotlib.dates interaction
* :ghissue:`782`: mplot3d: grid doesn't update after adding a slider to figure?
* :ghissue:`703`: pcolormesh help not helpful
* :ghissue:`1082`: doc: in pcolormesh docstring, say what it does.
* :ghissue:`1068`: Add stairstep plotting functionality
* :ghissue:`1078`: doc: note that IDLE doesn't work with interactive mode.
* :ghissue:`704`: ignore case in `edgecolors` keyword in `pcolormesh` (and possibly other places)
* :ghissue:`708`: set_clim not working with NonUniformImage
* :ghissue:`768`: Add "tight_layout" button to toolbar
* :ghissue:`791`: v1.1.1 release candidate testing
* :ghissue:`844`: imsave/imshow and cmaps
* :ghissue:`939`: test failure: matplotlib.tests.test_mathtext.mathfont_stix_14_test.test
* :ghissue:`875`: Replace "jet" with "hot" as the default colormap
* :ghissue:`881`: "Qualitative" colormaps represented as continuous
* :ghissue:`1072`: For a text artist, if it has a _bbox_patch associated with it, the conta...
* :ghissue:`1071`: patches.polygon: fix bug in handling of path closing, #1018.
* :ghissue:`1018`: BUG: check for closed path in Polygon.set_xy()
* :ghissue:`1066`: fix limit calculation of step* histogram
* :ghissue:`1073`: Mplot3d/input broadcast
* :ghissue:`906`: User-specified medians and conf. intervals in boxplots
* :ghissue:`899`: Update for building matplotlib under Mac OS X 10.7 Lion and XCode > 4.2
* :ghissue:`1057`: Contour norm scaling
* :ghissue:`1035`: Added a GTK3 implementation of the SubplotTool window.
* :ghissue:`807`: Crash when using zoom tools on a plot: AutoMinorLocator after MultipleLocator gives "ValueError: Need at least two major ticks to find minor tick locations"
* :ghissue:`1023`: New button to toolbar for tight_layout.
* :ghissue:`1056`: Test framework cleanups
* :ghissue:`778`: Make tests faster
* :ghissue:`1048`: some matplotlib examples incompatible with wxpython 2.9
* :ghissue:`1024`: broken links in the gallery
* :ghissue:`1054`:  stix_fonts_demo.py fails with bad refcount
* :ghissue:`960`: Fixed logformatting for non integer bases.
* :ghissue:`897`: GUI icon in Tkinter
* :ghissue:`1053`: Move Python 3 import of reload() to the module that uses it
* :ghissue:`1049`: Update examples/user_interfaces/embedding_in_wx2.py
* :ghissue:`1050`: Update examples/user_interfaces/embedding_in_wx4.py
* :ghissue:`1051`: Update examples/user_interfaces/mathtext_wx.py
* :ghissue:`1052`: Update examples/user_interfaces/wxcursor_demo.py
* :ghissue:`1047`: Enable building on Python 3.3 for Windows
* :ghissue:`819`: Add new plot style: stackplot
* :ghissue:`1036`: Move all figures to the front with a non-interactive show() in macosx backend.
* :ghissue:`1042`: Three more plot_directive configuration options
* :ghissue:`1044`: plots not being displayed in OSX 10.8
* :ghissue:`1022`: contour: map extended ranges to "under" and "over" values
* :ghissue:`1007`: modifying GTK3 example to use pygobject, and adding a simple example to demonstrate NavigationToolbar in GTK3
* :ghissue:`1004`: Added savefig.bbox option to matplotlibrc
* :ghissue:`976`: Fix embedding_in_qt4_wtoolbar.py on Python 3
* :ghissue:`1013`: compilation warnings in _macosx.m
* :ghissue:`1034`: MdH = allow compilation on recent Mac OS X without compiler warnings
* :ghissue:`964`: Animation clear_temp=False reuses old frames
* :ghissue:`1028`: Fix use() so that it is possible to reset the rcParam.
* :ghissue:`1033`: Py3k: reload->imp.reload
* :ghissue:`1002`: Fixed potential overflow exception in the lines.contains() method
* :ghissue:`1025`: Timers
* :ghissue:`989`: Animation subprocess bug
* :ghissue:`898`: Added warnings for easily confusible subplot/subplots invokations
* :ghissue:`963`: Add detection of file extension for file-like objects
* :ghissue:`973`: Fix sankey.py pep8 and py3 compatibility
* :ghissue:`972`: Force closing PIL image files
* :ghissue:`981`: Fix pathpatch3d_demo.py on Python 3
* :ghissue:`980`: Fix basic_units.py on Python 3. PEP8 and PyLint cleanup.
* :ghissue:`996`: macosx backend broken by #901: QuadMesh fails so colorbar fails
* :ghissue:`1017`: axes.Axes.step() function not documented
* :ghissue:`1014`: qt4: remove duplicate file save button; and remove trailing whitespace
* :ghissue:`655`: implement path_effects for Line2D objects
* :ghissue:`999`: pcolormesh edgecolor of "None"
* :ghissue:`1011`: fix for bug #996 and related issues
* :ghissue:`1009`: Simplify import statement
* :ghissue:`982`: Supported FreeBSD10 as per #225
* :ghissue:`225`: Add support for FreeBSD >6.x
* :ghissue:`985`: support current and future FreeBSD releases
* :ghissue:`1006`: MacOSX backend throws exception when plotting a quadmesh
* :ghissue:`1000`: Fix traceback for vlines/hlines, when an empty list or array passed in for x/y.
* :ghissue:`1001`: Bug fix for issue #955
* :ghissue:`994`: Fix bug in pcolorfast introduced by #901
* :ghissue:`993`: Fix typo
* :ghissue:`908`: use Qt window title as default savefig filename
* :ghissue:`830`: standard key for closing figure ("q")
* :ghissue:`971`: Close fd temp file following rec2csv_bad_shape test
* :ghissue:`851`: Simple GUI interface enhancements
* :ghissue:`979`: Fix test_mouseclicks.py on Python 3
* :ghissue:`977`: Fix lasso_selector_demo.py on Python 3
* :ghissue:`970`: Fix tiff and jpeg export via PIL
* :ghissue:`707`: key_press_event in pyqt4 embedded matplotlib
* :ghissue:`243`: Debug version/symbols for win32
* :ghissue:`255`: Classes in _transforms.h in global namespace
* :ghissue:`961`: Issue 807 auto minor locator
* :ghissue:`345`: string symbol markers ("scattertext" plot)
* :ghissue:`247`: DLL load failed
* :ghissue:`808`: pip install matplotlib fails
* :ghissue:`168`: setupext.py incorrect for Mac OS X
* :ghissue:`213`: Fixing library path in setupext.py for Mac
