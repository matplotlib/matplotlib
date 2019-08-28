==========
Installing
==========

.. note::

    If you wish to contribute to the project, it's recommended you
    :ref:`install the latest development version<install_from_source>`.

.. contents::

Installing an official release
==============================

Matplotlib and its dependencies are available as wheel packages for macOS,
Windows and Linux distributions::

  python -m pip install -U pip
  python -m pip install -U matplotlib

.. note::

   The following backends work out of the box: Agg, ps, pdf, svg and TkAgg.

   For support of other GUI frameworks, LaTeX rendering, saving
   animations and a larger selection of file formats, you may need to
   install :ref:`additional dependencies <install_requirements>`.

Although not required, we suggest also installing ``IPython`` for
interactive use.  To easily install a complete Scientific Python
stack, see :ref:`install_scipy_dists` below.

Test data
---------

The wheels (:file:`*.whl`) on the `PyPI download page
<https://pypi.org/project/matplotlib/>`_ do not contain test data
or example code.

If you want to try the many demos that come in the Matplotlib source
distribution, download the :file:`*.tar.gz` file and look in the
:file:`examples` subdirectory.

To run the test suite:

* extract the :file:`lib/matplotlib/tests` or :file:`lib/mpl_toolkits/tests`
  directories from the source distribution;
* install test dependencies: `pytest <https://pypi.org/project/pytest>`_,
  Pillow, MiKTeX, GhostScript, ffmpeg, avconv, ImageMagick, and `Inkscape
  <https://inkscape.org/>`_;
* run ``python -mpytest``.

Third-party distributions of Matplotlib
=======================================

.. _install_scipy_dists:

Scientific Python Distributions
-------------------------------

`Anaconda <https://www.anaconda.com/>`_ and `Canopy
<https://www.enthought.com/products/canopy/>`_ and `ActiveState
<https://www.activestate.com/activepython/downloads>`_ are excellent
choices that "just work" out of the box for Windows, macOS and common
Linux platforms. `WinPython <https://winpython.github.io/>`_ is an
option for Windows users.  All of these distributions include
Matplotlib and *lots* of other useful (data) science tools.

Linux: using your package manager
---------------------------------

If you are on Linux, you might prefer to use your package manager.  Matplotlib
is packaged for almost every major Linux distribution.

* Debian / Ubuntu: ``sudo apt-get install python3-matplotlib``
* Fedora: ``sudo dnf install python3-matplotlib``
* Red Hat: ``sudo yum install python3-matplotlib``
* Arch: ``sudo pacman -S python-matplotlib``

.. _install_from_source:

Installing from source
======================

If you are interested in contributing to Matplotlib development,
running the latest source code, or just like to build everything
yourself, it is not difficult to build Matplotlib from source.  Grab
the latest *tar.gz* release file from `the PyPI files page
<https://pypi.org/project/matplotlib/>`_, or if you want to
develop Matplotlib or just need the latest bugfixed version, grab the
latest git version, and see :ref:`install-from-git`.

The standard environment variables :envvar:`CC`, :envvar:`CXX`,
:envvar:`PKG_CONFIG` are respected.  This means you can set them if your
toolchain is prefixed. This may be used for cross compiling. ::

  export CC=x86_64-pc-linux-gnu-gcc
  export CXX=x86_64-pc-linux-gnu-g++
  export PKG_CONFIG=x86_64-pc-linux-gnu-pkg-config

Once you have satisfied the requirements detailed below (mainly
Python, NumPy, libpng and FreeType), you can build Matplotlib.
::

  cd matplotlib
  python -mpip install .

We provide a setup.cfg_ file which you can use to customize the build
process. For example, which default backend to use, whether some of the
optional libraries that Matplotlib ships with are installed, and so on.  This
file will be particularly useful to those packaging Matplotlib.

.. _setup.cfg: https://raw.githubusercontent.com/matplotlib/matplotlib/master/setup.cfg.template

If you have installed prerequisites to nonstandard places and need to
inform Matplotlib where they are, edit ``setupext.py`` and add the base
dirs to the ``basedir`` dictionary entry for your ``sys.platform``;
e.g., if the header of some required library is in
``/some/path/include/someheader.h``, put ``/some/path`` in the
``basedir`` list for your platform.

.. _install_requirements:

Dependencies
------------

Matplotlib requires the following dependencies:

* `Python <https://www.python.org/downloads/>`_ (>= 3.6)
* `FreeType <https://www.freetype.org/>`_ (>= 2.3)
* `libpng <http://www.libpng.org>`_ (>= 1.2)
* `NumPy <http://www.numpy.org>`_ (>= 1.11)
* `setuptools <https://setuptools.readthedocs.io/en/latest/>`_
* `cycler <http://matplotlib.org/cycler/>`_ (>= 0.10.0)
* `dateutil <https://pypi.org/project/python-dateutil>`_ (>= 2.1)
* `kiwisolver <https://github.com/nucleic/kiwi>`_ (>= 1.0.0)
* `pyparsing <https://pyparsing.wikispaces.com/>`_

Optionally, you can also install a number of packages to enable better user
interface toolkits. See :ref:`what-is-a-backend` for more details on the
optional Matplotlib backends and the capabilities they provide.

* `Tk <https://docs.python.org/3/library/tk.html>`_ (>= 8.3, != 8.6.0 or
  8.6.1): for the Tk-based backends;
* `PyQt4 <https://pypi.org/project/PyQt4>`_ (>= 4.6) or
  `PySide <https://pypi.org/project/PySide>`_ (>= 1.0.3) [#]_: for the
  Qt4-based backends;
* `PyQt5 <https://pypi.org/project/PyQt5>`_: for the Qt5-based backends;
* `PyGObject <https://pypi.org/project/PyGObject/>`_: for the GTK3-based
  backends [#]_;
* `wxPython <https://www.wxpython.org/>`_ (>= 4) [#]_: for the wx-based
  backends;
* `cairocffi <https://cairocffi.readthedocs.io/en/latest/>`_ (>= 0.8) or
  `pycairo <https://pypi.org/project/pycairo>`_: for the cairo-based
  backends;
* `Tornado <https://pypi.org/project/tornado>`_: for the WebAgg backend;

.. [#] PySide cannot be pip-installed on Linux (but can be conda-installed).
.. [#] If using pip (and not conda), PyGObject must be built from source; see
       https://pygobject.readthedocs.io/en/latest/devguide/dev_environ.html.
.. [#] If using pip (and not conda) on Linux, wxPython wheels must be manually
       downloaded from https://wxpython.org/pages/downloads/.

For better support of animation output format and image file formats, LaTeX,
etc., you can install the following:

* `ffmpeg <https://www.ffmpeg.org/>`_/`avconv
  <https://libav.org/avconv.html>`_: for saving movies;
* `ImageMagick <https://www.imagemagick.org/script/index.php>`_: for saving
  animated gifs;
* `Pillow <https://pillow.readthedocs.io/en/latest/>`_ (>= 3.4): for a larger
  selection of image file formats: JPEG, BMP, and TIFF image files;
* `LaTeX <https://miktex.org/>`_ and `GhostScript (>=9.0)
  <https://ghostscript.com/download/>`_ : for rendering text with LaTeX.

.. note::

   Matplotlib depends on non-Python libraries.

   On Linux and OSX, pkg-config_ can be used to find required non-Python
   libraries and thus make the install go more smoothly if the libraries and
   headers are not in the expected locations.

   .. _pkg-config: https://www.freedesktop.org/wiki/Software/pkg-config/

   If not using pkg-config (in particular on Windows), you may need to set the
   include path (to the FreeType, libpng, and zlib headers) and link path (to
   the FreeType, libpng, and zlib libraries) explicitly, if they are not in
   standard locations.  This can be done using standard environment variables
   -- on Linux and OSX:

   .. code-block:: sh

      export CFLAGS='-I/directory/containing/ft2build.h ...'
      export LDFLAGS='-L/directory/containing/libfreetype.so ...'

   and on Windows:

   .. code-block:: bat

      set CL=/IC:\directory\containing\ft2build.h ...
      set LINK=/LIBPATH:C:\directory\containing\freetype.lib ...

   where ``...`` means "also give, in the same format, the directories
   containing ``png.h`` and ``zlib.h`` for the include path, and for
   ``libpng.so``/``png.lib`` and ``libz.so``/``z.lib`` for the link path."

.. note::

  The following libraries are shipped with Matplotlib:

  - ``Agg``: the Anti-Grain Geometry C++ rendering engine;
  - ``qhull``: to compute Delaunay triangulation;
  - ``ttconv``: a TrueType font utility.

.. _build_linux:

Building on Linux
-----------------

It is easiest to use your system package manager to install the dependencies.

If you are on Debian/Ubuntu, you can get all the dependencies
required to build Matplotlib with::

   sudo apt-get build-dep python-matplotlib

If you are on Fedora, you can get all the dependencies required to build
Matplotlib with::

   sudo dnf builddep python-matplotlib

If you are on RedHat, you can get all the dependencies required to build
Matplotlib by first installing ``yum-builddep`` and then running::

   su -c "yum-builddep python-matplotlib"

These commands do not build Matplotlib, but instead get and install the
build dependencies, which will make building from source easier.

.. _build_osx:

Building on macOS
-----------------

The build situation on macOS is complicated by the various places one
can get the libpng and FreeType requirements (MacPorts, Fink,
/usr/X11R6), the different architectures (e.g., x86, ppc, universal), and
the different macOS versions (e.g., 10.4 and 10.5). We recommend that you build
the way we do for the macOS release: get the source from the tarball or the
git repository and install the required dependencies through a third-party
package manager. Two widely used package managers are Homebrew, and MacPorts.
The following example illustrates how to install libpng and FreeType using
``brew``::

  brew install libpng freetype pkg-config

If you are using MacPorts, execute the following instead::

  port install libpng freetype pkgconfig

After installing the above requirements, install Matplotlib from source by
executing::

  python -mpip install .

Note that your environment is somewhat important. Some conda users have
found that, to run the tests, their PYTHONPATH must include
/path/to/anaconda/.../site-packages and their DYLD_FALLBACK_LIBRARY_PATH
must include /path/to/anaconda/lib.

.. _build_windows:

Building on Windows
-------------------

The Python shipped from https://www.python.org is compiled with Visual Studio
2015 for 3.5+.  Python extensions should be compiled with the same
compiler, see e.g.
https://packaging.python.org/guides/packaging-binary-extensions/#setting-up-a-build-environment-on-windows
for how to set up a build environment.

Since there is no canonical Windows package manager, the methods for building
FreeType, zlib, and libpng from source code are documented as a build script
at `matplotlib-winbuild <https://github.com/jbmohler/matplotlib-winbuild>`_.

There are a few possibilities to build Matplotlib on Windows:

* Wheels via `matplotlib-winbuild <https://github.com/jbmohler/matplotlib-winbuild>`_
* Wheels by using conda packages (see below)
* Conda packages (see below)

If you are building your own Matplotlib wheels (or sdists), note that any DLLs
that you copy into the source tree will be packaged too.

Wheel builds using conda packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a wheel build, but we use conda packages to get all the requirements.
The binary requirements (png, FreeType,...) are statically linked and therefore
not needed during the wheel install.

Set up the conda environment. Note, if you want a qt backend, add ``pyqt`` to
the list of conda packages.

::

  conda create -n "matplotlib_build" python=3.7 numpy python-dateutil pyparsing tornado cycler tk libpng zlib freetype msinttypes
  conda activate matplotlib_build

For building, call the script ``build_alllocal.cmd`` in the root folder of the
repository::

  build_alllocal.cmd

Conda packages
^^^^^^^^^^^^^^

The conda packaging scripts for Matplotlib are available at
https://github.com/conda-forge/matplotlib-feedstock.


.. _installing-faq:

FAQ
===

.. contents::
   :backlinks: none

Report a compilation problem
----------------------------

See :ref:`reporting-problems`.

Matplotlib compiled fine, but nothing shows up when I use it
------------------------------------------------------------

The first thing to try is a :ref:`clean install <clean-install>` and see if
that helps.  If not, the best way to test your install is by running a script,
rather than working interactively from a python shell or an integrated
development environment such as :program:`IDLE` which add additional
complexities. Open up a UNIX shell or a DOS command prompt and run, for
example::

   python -c "from pylab import *; plot(); show()" --verbose-helpful

This will give you additional information about which backends Matplotlib is
loading, version information, and more. At this point you might want to make
sure you understand Matplotlib's :doc:`configuration </tutorials/introductory/customizing>`
process, governed by the :file:`matplotlibrc` configuration file which contains
instructions within and the concept of the Matplotlib backend.

If you are still having trouble, see :ref:`reporting-problems`.

.. _clean-install:

How to completely remove Matplotlib
-----------------------------------

Occasionally, problems with Matplotlib can be solved with a clean
installation of the package.  In order to fully remove an installed Matplotlib:

1. Delete the caches from your :ref:`Matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Delete any Matplotlib directories or eggs from your :ref:`installation
   directory <locating-matplotlib-install>`.

Linux Notes
-----------

To install Matplotlib at the system-level, we recommend that you use your
distribution's package manager.  This will guarantee that Matplotlib's
dependencies will be installed as well.

If, for some reason, you cannot use the package manager, you may use the wheels
available on PyPI::

   python -mpip install matplotlib

or :ref:`build Matplotlib from source <install-from-git>`.

OSX Notes
=========

.. _which-python-for-osx:

Which python for OSX?
---------------------

Apple ships OSX with its own Python, in ``/usr/bin/python``, and its own copy
of Matplotlib. Unfortunately, the way Apple currently installs its own copies
of NumPy, Scipy and Matplotlib means that these packages are difficult to
upgrade (see `system python packages`_).  For that reason we strongly suggest
that you install a fresh version of Python and use that as the basis for
installing libraries such as NumPy and Matplotlib.  One convenient way to
install Matplotlib with other useful Python software is to use one of the
excellent Python scientific software collections that are now available:

.. _system python packages:
    https://github.com/MacPython/wiki/wiki/Which-Python#system-python-and-extra-python-packages

- Anaconda_ from `Continuum Analytics`_
- Canopy_ from Enthought_

.. _Canopy: https://www.enthought.com/products/canopy/
.. _Anaconda: https://www.anaconda.com/
.. _Enthought: https://www.enthought.com
.. _Continuum Analytics: https://www.continuum.io

These collections include Python itself and a wide range of libraries; if you
need a library that is not available from the collection, you can install it
yourself using standard methods such as *pip*.  Continuum and Enthought offer
their own installation support for these collections; see the Ananconda and
Canopy web pages for more information.

Other options for a fresh Python install are the standard installer from
`python.org <https://www.python.org/downloads/mac-osx/>`_, or installing
Python using a general OSX package management system such as `homebrew
<http://brew.sh>`_ or `macports <https://www.macports.org>`_.  Power users on
OSX will likely want one of homebrew or macports on their system to install
open source software packages, but it is perfectly possible to use these
systems with another source for your Python binary, such as Anaconda, Canopy
or Python.org Python.

.. _install_osx_binaries:

Installing OSX binary wheels
----------------------------

If you are using Python from https://www.python.org, Homebrew, or Macports,
then you can use the standard pip installer to install Matplotlib binaries in
the form of wheels.

pip is installed by default with python.org and Homebrew Python, but needs to
be manually installed on Macports with ::

   sudo port install py36-pip

Once pip is installed, you can install Matplotlib and all its dependencies with
from the Terminal.app command line::

   python3 -mpip install matplotlib

(``sudo python3.6 ...`` on Macports).

You might also want to install IPython or the Jupyter notebook (``python3 -mpip
install ipython notebook``).

pip problems
^^^^^^^^^^^^

If you get errors with pip trying to run a compiler like ``gcc`` or ``clang``,
then the first thing to try is to `install xcode
<https://guide.macports.org/chunked/installing.html#installing.xcode>`_ and
retry the install.  If that does not work, then check
:ref:`reporting-problems`.

Checking your installation
--------------------------

The new version of Matplotlib should now be on your Python "path".  Check this
at the Terminal.app command line::

  python3 -c 'import matplotlib; print(matplotlib.__version__, matplotlib.__file__)'

You should see something like ::

  3.0.0 /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/__init__.py

where ``3.0.0`` is the Matplotlib version you just installed, and the path
following depends on whether you are using Python.org Python, Homebrew or
Macports.  If you see another version, or you get an error like ::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then check that the Python binary is the one you expected by running ::

  which python3

If you get a result like ``/usr/bin/python...``, then you are getting the
Python installed with OSX, which is probably not what you want.  Try closing
and restarting Terminal.app before running the check again. If that doesn't fix
the problem, depending on which Python you wanted to use, consider reinstalling
Python.org Python, or check your homebrew or macports setup.  Remember that
the disk image installer only works for Python.org Python, and will not get
picked up by other Pythons.  If all these fail, please :ref:`let us know
<reporting-problems>`.

.. _install-from-git:

Install from source
-------------------

Clone the main source using one of::

   git clone git@github.com:matplotlib/matplotlib.git

or::

   git clone git://github.com/matplotlib/matplotlib.git

and build and install as usual with::

   cd matplotlib
   python -mpip install .

.. note::

   If you are on Debian/Ubuntu, you can get all the dependencies required to
   build Matplotlib with::

      sudo apt-get build-dep python-matplotlib

   If you are on Fedora/RedHat, you can get all the dependencies required to
   build Matplotlib by first installing ``yum-builddep`` and then running::

      su -c 'yum-builddep python-matplotlib'

   This does not build Matplotlib, but it does get all of the build
   dependencies, which will make building from source easier.

If you want to be able to follow the development branch as it changes
just replace the last step with::

   python -mpip install -e .

This creates links and installs the command line script in the appropriate
places.

.. note::
   OSX users please see the :ref:`build_osx` guide.

   Windows users please see the :ref:`build_windows` guide.

Then, if you want to update your Matplotlib at any time, just do::

   git pull

When you run ``git pull``, if the output shows that only Python files have
been updated, you are all set. If C files have changed, you need to run ``pip
install -e .`` again to compile them.

There is more information on :ref:`using git <using-git>` in the developer
docs.
