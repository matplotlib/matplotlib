.. _credits:

*******
Credits
*******


matplotlib was written by John Hunter and is now developed and
maintained by a number of `active
<http://www.ohloh.net/projects/matplotlib/contributors>`_ developers.
The current lead developer of matplotlib is Michael Droettboom.

Special thanks to those who have made valuable contributions (roughly
in order of first contribution by date).  Any list like this is bound
to be incomplete and can't capture the thousands and thousands of
contributions over the years from these and others:

Jeremy O'Donoghue
  wrote the wx backend

Andrew Straw
  Provided much of the log scaling architecture, the fill command, PIL
  support for imshow, and provided many examples.  He also wrote the
  support for dropped axis spines and the `buildbot
  <http://mpl-buildbot.code.astraw.com/>`_ unit testing infrastructure
  which triggers the JPL/James Evans platform specific builds and
  regression test image comparisons from svn matplotlib across
  platforms on svn commits.

Charles Twardy
  provided the impetus code for the legend class and has made
  countless bug reports and suggestions for improvement.

Gary Ruben
  made many enhancements to errorbar to support x and y
  errorbar plots, and added a number of new marker types to plot.

John Gill
  wrote the table class and examples, helped with support for
  auto-legend placement, and added support for legending scatter
  plots.

David Moore
  wrote the paint backend (no longer used)

Todd Miller
  supported by `STSCI <http://www.stsci.edu>`_ contributed the TkAgg
  backend and the numerix module, which allows matplotlib to work with
  either numeric or numarray.  He also ported image support to the
  postscript backend, with much pain and suffering.

Paul Barrett
  supported by `STSCI <http://www.stsci.edu>`_ overhauled font
  management to provide an improved, free-standing, platform
  independent font manager with a WC3 compliant font finder and cache
  mechanism and ported truetype and mathtext to PS.

Perry Greenfield
  supported by `STSCI <http://www.stsci.edu>`_ overhauled and
  modernized the goals and priorities page, implemented an improved
  colormap framework, and has provided many suggestions and a lot of
  insight to the overall design and organization of matplotlib.

Jared Wahlstrand
  wrote the initial SVG backend.

Steve Chaplin
  served as the GTK maintainer and wrote the Cairo and
  GTKCairo backends.

Jim Benson
  provided the patch to handle vertical mathttext.

Gregory Lielens
  provided the FltkAgg backend and several patches for the frontend,
  including contributions to toolbar2, and support for log ticking
  with alternate bases and major and minor log ticking.

Darren Dale

  did the work to do mathtext exponential labeling for log plots,
  added improved support for scalar formatting, and did the lions
  share of the `psfrag
  <http://www.ctan.org/tex-archive/help/Catalogue/entries/psfrag.html?action=/tex-archive/macros/latex/contrib/supported/psfrag>`_
  LaTeX support for postscript. He has made substantial contributions
  to extending and maintaining the PS and Qt backends, and wrote the
  site.cfg and matplotlib.conf build and runtime configuration
  support.  He setup the infrastructure for the sphinx documentation
  that powers the mpl docs.

Paul Mcguire
  provided the pyparsing module on which mathtext relies, and made a
  number of optimizations to the matplotlib mathtext grammar.


Fernando Perez
  has provided numerous bug reports and patches for cleaning up
  backend imports and expanding pylab functionality, and provided
  matplotlib support in the pylab mode for `ipython
  <http://ipython.org>`_.  He also provided the
  :func:`~matplotlib.pyplot.matshow` command, and wrote TConfig, which
  is the basis for the experimental traited mpl configuration.

Andrew Dalke
  of `Dalke Scientific Software <http://www.dalkescientific.com/>`_ contributed the
  strftime formatting code to handle years earlier than 1900.

Jochen Voss
  served as PS backend maintainer and has contributed several
  bugfixes.

Nadia Dencheva

  supported by `STSCI <http://www.stsci.edu>`_ provided the contouring and
  contour labeling code.

Baptiste Carvello
  provided the key ideas in a patch for proper
  shared axes support that underlies ganged plots and multiscale
  plots.

Jeffrey Whitaker
  at `NOAA <http://www.boulder.noaa.gov>`_ wrote the
  :ref:`toolkit_basemap` toolkit

Sigve Tjoraand, Ted Drain, James Evans
  and colleagues at the `JPL <http://www.jpl.nasa.gov>`_ collaborated
  on the QtAgg backend and sponsored development of a number of
  features including custom unit types, datetime support, scale free
  ellipses, broken bar plots and more.  The JPL team wrote the unit
  testing image comparison `infrastructure
  <https://github.com/matplotlib/matplotlib/tree/master/test>`_
  for regression test image comparisons.

James Amundson
  did the initial work porting the qt backend to qt4

Eric Firing
  has contributed significantly to contouring, masked
  array, pcolor, image and quiver support, in addition to ongoing
  support and enhancements in performance, design and code quality in
  most aspects of matplotlib.

Daishi Harada
  added support for "Dashed Text".  See `dashpointlabel.py
  <examples/pylab_examples/dashpointlabel.py>`_ and
  :class:`~matplotlib.text.TextWithDash`.

Nicolas Young
  added support for byte images to imshow, which are
  more efficient in CPU and memory, and added support for irregularly
  sampled images.

The `brainvisa <http://brainvisa.info>`_ Orsay team and Fernando Perez
  added Qt support to `ipython <http://ipython.org>`_ in pylab mode.


Charlie Moad
  contributed work to matplotlib's Cocoa support and has done a lot of work on the OSX and win32 binary releases.

Jouni K. Seppänen
  wrote the PDF backend and contributed numerous
  fixes to the code, to tex support and to the get_sample_data handler

Paul Kienzle
  improved the picking infrastructure for interactive plots, and with
  Alex Mont contributed fast rendering code for quadrilateral meshes.

Michael Droettboom
  supported by `STSCI <http://www.stsci.edu>`_ wrote the enhanced
  mathtext support, implementing Knuth's box layout algorithms, saving
  to file-like objects across backends, and is responsible for
  numerous bug-fixes, much better font and unicode support, and
  feature and performance enhancements across the matplotlib code
  base. He also rewrote the transformation infrastructure to support
  custom projections and scales.

John Porter, Jonathon Taylor and Reinier Heeres
  John Porter wrote the mplot3d module for basic 3D plotting in
  matplotlib, and Jonathon Taylor and Reinier Heeres ported it to the
  refactored transform trunk.

Jae-Joon Lee
  Implemented fancy arrows and boxes, rewrote the legend
  support to handle multiple columns and fancy text boxes, wrote the
  axes grid toolkit, and has made numerous contributions to the code
  and documentation

Paul Ivanov
  Has worked on getting matplotlib integrated better with other tools,
  such as Sage and IPython, and getting the test infrastructure
  faster, lighter and meaner.  Listen to his podcast.

Tony Yu
  Has been involved in matplotlib since the early days, and recently
  has contributed stream plotting among many other improvements.  He
  is the author of mpltools.

Michiel de Hoon
  Wrote and maintains the macosx backend.

Ian Thomas
  Contributed, among other things, the triangulation (tricolor and
  tripcontour) methods.

Benjamin Root
  Has significantly improved the capabilities of the 3D plotting.  He
  has improved matplotlib's documentation and code quality throughout,
  and does invaluable triaging of pull requests and bugs.

Phil Elson
  Fixed some deep-seated bugs in the transforms framework, and has
  been laser-focused on improving polish throughout matplotlib,
  tackling things that have been considered to large and daunting for
  a long time.

Damon McDougall
  Added triangulated 3D surfaces and stack plots to matplotlib.
