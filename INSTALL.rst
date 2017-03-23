.. The source of this document is INSTALL.rst. During the doc build process,
.. this file is copied over to doc/users/installing.rst.
.. Therefore, you must edit INSTALL.rst, *not* doc/users/installing.rst!

.. _pip: https://pypi.python.org/pypi/pip/

==========
Installing
==========

.. note::

    If you wish to contribute to the project, it's recommended you
    :ref:`install the latest development version<install_from_source>`.


.. contents::

Installing an official release
==============================

Matplotlib and most of its dependencies are all available as wheel packages for
macOS, Windows and Linux distributions::

  pip install -U matplotlib


Third-party distributions of Matplotlib
=======================================

Scientific Python distributions: Conda, Canopy...
-------------------------------------------------

The first option is to use one of the pre-packaged Python distributions that
already provide Matplotlib built-in.  Both `Anaconda
<https://www.continuum.io/downloads/>`_ and `Canopy
<https://www.enthought.com/products/canopy/>`_ are both excellent choices that
"just work" out of the box for Windows, macOS and common Linux platforms.  Both
of these distributions include Matplotlib and *lots* of other useful tools.

Linux : using your package manager
----------------------------------

If you are on Linux, you might prefer to use your package manager.  Matplotlib
is packaged for almost every major Linux distribution.

* Debian / Ubuntu: ``sudo apt-get install python-matplotlib``
* Fedora: ``sudo dnf install python-matplotlib``
* Red Hat: ``sudo yum install python-matplotlib``
* Arch: ``sudo pacman -S python-matplotlib``

.. _installing_windows:

Windows
-------

We strongly recommend using `SciPy-stack compatible Python distributions
<https://www.scipy.org/install.html>`_ such as WinPython, Python(x,y),
Enthought Canopy, or Continuum Anaconda, which have Matplotlib and its
dependencies, plus other useful packages, preinstalled.

For `standard Python <https://www.python.org/downloads/>`_ installations,
install Matplotlib using pip_::

    python -m pip install -U pip setuptools
    python -m pip install matplotlib

In case Python 2.7 or 3.4 are not installed for all users,
the Microsoft Visual C++ 2008
(`64 bit <https://www.microsoft.com/en-us/download/details.aspx?id=15336>`__
or
`32 bit <https://www.microsoft.com/en-us/download/details.aspx?id=29>`__
for Python 2.7) or Microsoft Visual C++ 2010
(`64 bit <https://www.microsoft.com/en-us/download/details.aspx?id=14632>`__
or
`32 bit <https://www.microsoft.com/en-us/download/details.aspx?id=5555>`__
for Python 3.4) redistributable packages need to be installed.


The Windows wheels (:file:`*.whl`) on the `PyPI download page
<https://pypi.python.org/pypi/matplotlib/>`_ do not contain test data
or example code.
If you want to try the many demos that come in the Matplotlib source
distribution, download the :file:`*.tar.gz` file and look in the
:file:`examples` subdirectory.
To run the test suite:

 * extract the :file:`lib\\matplotlib\\tests` or
   :file:`lib\\mpl_toolkits\\tests` directories from the source distribution;
 * install test dependencies: `pytest <https://pypi.python.org/pypi/pytest>`_,
   `mock <https://pypi.python.org/pypi/mock>`_, Pillow, MiKTeX, GhostScript,
   ffmpeg, avconv, mencoder, ImageMagick, and `Inkscape
   <https://inkscape.org/>`_;
 * run ``py.test path\to\tests\directory``.

.. note::

   The following backends work out of the box: Agg, TkAgg, ps, pdf and svg.
   TkAgg is probably the best backend for interactive use from the standard
   Python shell or from IPython and is enabled as default.

   GTK3 is not supported on Windows.

   For support for other backends, LaTeX rendering, animation input/output and
   a larger selection of file formats, you may need to install :ref:`additional
   dependencies <install_requirements>`.


.. _install_from_source:

Installing from source
======================

If you are interested in contributing to Matplotlib development,
running the latest source code, or just like to build everything
yourself, it is not difficult to build Matplotlib from source.  Grab
the latest *tar.gz* release file from `the PyPI files page
<https://pypi.python.org/pypi/matplotlib/>`_, or if you want to
develop Matplotlib or just need the latest bugfixed version, grab the
latest git version :ref:`install-from-git`.

The standard environment variables `CC`, `CXX`, `PKG_CONFIG` are respected.
This means you can set them if your toolchain is prefixed. This may be used for
cross compiling.
::

  export CC=x86_64-pc-linux-gnu-gcc
  export CXX=x86_64-pc-linux-gnu-g++
  export PKG_CONFIG=x86_64-pc-linux-gnu-pkg-config

Once you have satisfied the requirements detailed below (mainly
Python, NumPy, libpng and FreeType), you can build Matplotlib.
::

  cd matplotlib
  python setup.py build
  python setup.py install

We provide a `setup.cfg
<https://raw.githubusercontent.com/matplotlib/matplotlib/master/setup.cfg.template>`_
file that goes with :file:`setup.py` which you can use to customize
the build process. For example, which default backend to use, whether
some of the optional libraries that Matplotlib ships with are
installed, and so on.  This file will be particularly useful to those
packaging Matplotlib.

If you have installed prerequisites to nonstandard places and need to
inform Matplotlib where they are, edit ``setupext.py`` and add the base
dirs to the ``basedir`` dictionary entry for your ``sys.platform``;
e.g., if the header of some required library is in
``/some/path/include/someheader.h``, put ``/some/path`` in the
``basedir`` list for your platform.

.. _install_requirements:

Dependencies
------------

Matplotlib requires a large number of dependencies:

  * `Python <https://www.python.org/downloads/>`_ (>= 2.7 or >= 3.4)
  * `NumPy <http://www.numpy.org>`_ (>= |minimum_numpy_version|)
  * `setuptools <https://setuptools.readthedocs.io/en/latest/>`__
  * dateutil (>= 1.1)
  * `pyparsing <https://pyparsing.wikispaces.com/>`__
  * `libpng <http://www.libpng.org>`__ (>= 1.2)
  * `pytz <http://pytz.sourceforge.net/>`__
  * FreeType (>= 2.3)
  * `cycler <http://matplotlib.org/cycler/>`__ (>= 0.10.0)
  * `six <https://pypi.python.org/pypi/six>`_
  * `backports.functools_lru_cache <https://pypi.python.org/pypi/backports.functools_lru_cache>`_
    (for Python 2.7 only)
  * `subprocess32 <https://pypi.python.org/pypi/subprocess32/>`_ (for Python
    2.7 only, on Linux and macOS only)

Optionally, you can also install a number of packages to enable better user
interface toolkits. See :ref:`what-is-a-backend` for more details on the
optional Matplotlib backends and the capabilities they provide.

  * :term:`tk` (>= 8.3, != 8.6.0 or 8.6.1): for the TkAgg backend;
  * `PyQt4 <https://pypi.python.org/pypi/PyQt4>`_ (>= 4.4) or
    `PySide <https://pypi.python.org/pypi/PySide>`_: for the Qt4Agg backend;
  * `PyQt5 <https://pypi.python.org/pypi/PyQt5>`_: for the Qt5Agg backend;
  * :term:`pygtk` (>= 2.4): for the GTK and the GTKAgg backend;
  * :term:`wxpython` (>= 2.8 or later): for the WX or WXAgg backend;
  * `pycairo <https://pypi.python.org/pypi/pycairo>`_: for GTK3Cairo;
  * `Tornado <https://pypi.python.org/pypi/tornado>`_: for the WebAgg backend.

For better support of animation output format and image file formats, LaTeX,
etc., you can install the following:

  * `ffmpeg <https://www.ffmpeg.org/>`__/`avconv
    <https://libav.org/avconv.html>`__ or `mencoder
    <https://mplayerhq.hu/design7/news.html>`__ (for saving movies);
  * `ImageMagick <https://www.imagemagick.org/script/index.php>`__ (for saving
    animated gifs);
  * `Pillow <https://python-pillow.org/>`__ (for a larger selection of image
    file formats: JPEG, BMP, and TIFF image files);
  * `LaTeX <https://miktex.org/>`_ and `GhostScript <https://ghostscript.com/download/>`_
    (for rendering text with LaTeX);

.. note::

   Matplotlib depends on a large number of non-Python libraries.
   `pkg-config <https://www.freedesktop.org/wiki/Software/pkg-config/>`__
   can be used to find required non-Python libraries and thus make the install
   go more smoothly if the libraries and headers are not in the expected
   locations.

.. note::

  The following libraries are shipped with Matplotlib:

    - `Agg`: the Anti-Grain Geometry C++ rendering engine;
    - `qhull`: to compute Delaunay triangulation;
    - `ttconv`: a true type font utility.

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

  python setup.py install

Note that your environment is somewhat important. Some conda users have
found that, to run the tests, their PYTHONPATH must include
/path/to/anaconda/.../site-packages and their DYLD_FALLBACK_LIBRARY_PATH
must include /path/to/anaconda/lib.


.. _build_windows:

Building on Windows
-------------------

The Python shipped from https://www.python.org is compiled with Visual Studio
2008 for versions before 3.3, Visual Studio 2010 for 3.3 and 3.4, and
Visual Studio 2015 for 3.5 and 3.6.  Python extensions are recommended to be compiled
with the same compiler.

Since there is no canonical Windows package manager, the methods for building
FreeType, zlib, and libpng from source code are documented as a build script
at `matplotlib-winbuild <https://github.com/jbmohler/matplotlib-winbuild>`_.


There are a few possibilities to build Matplotlib on Windows:

* Wheels via `matplotlib-winbuild <https://github.com/jbmohler/matplotlib-winbuild>`_
* Wheels by using conda packages
* Conda packages

Wheel builds using conda packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a wheel build, but we use conda packages to get all the requirements. The binary
requirements (png, FreeType,...) are statically linked and therefore not needed during the wheel
install.

The commands below assume that you can compile a native Python lib for the Python version of your
choice. See `this howto <https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/>`_
for how to install and setup such environments. If in doubt: use Python >= 3.5 as it mostly works
without fiddling with environment variables::

  # create a new environment with the required packages
  conda create  -n "matplotlib_build" python=3.5 numpy python-dateutil pyparsing pytz tornado "cycler>=0.10" tk libpng zlib freetype
  activate matplotlib_build
  # if you want a qt backend, you also have to install pyqt (be aware that pyqt doesn't mix well if
  # you have created the environment with conda-forge already activated...)
  conda install pyqt
  # this package is only available in the conda-forge channel
  conda install -c conda-forge msinttypes
  # for Python 2.7
  conda install -c conda-forge backports.functools_lru_cache

  # copy the libs which have "wrong" names
  set LIBRARY_LIB=%CONDA_DEFAULT_ENV%\Library\lib
  mkdir lib || cmd /c "exit /b 0"
  copy %LIBRARY_LIB%\zlibstatic.lib lib\z.lib
  copy %LIBRARY_LIB%\libpng_static.lib lib\png.lib

  # Make the header files and the rest of the static libs available during the build
  # CONDA_DEFAULT_ENV is a env variable which is set to the currently active environment path
  set MPLBASEDIRLIST=%CONDA_DEFAULT_ENV%\Library\;.

  # build the wheel
  python setup.py bdist_wheel

The `build_alllocal.cmd` script in the root folder automates these steps if
you have already created and activated the conda environment.


Conda packages
^^^^^^^^^^^^^^

This needs a `working installed C compiler
<https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/>`_
for the version of Python you are compiling the package for but you don't need
to setup the environment variables::

  # only the first time...
  conda install conda-build

  # the Python version you want a package for...
  set CONDA_PY=3.5

  # builds the package, using a clean build environment
  conda build ci\conda_recipe

  # install the new package
  conda install --use-local matplotlib
