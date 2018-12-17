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

macOS
-----

To use the native OSX backend you will need :ref:`a framework build
<osxframework-faq>` build of Python.

Test data
---------

The wheels (:file:`*.whl`) on the `PyPI download page
<https://pypi.python.org/pypi/matplotlib/>`_ do not contain test data
or example code.

If you want to try the many demos that come in the Matplotlib source
distribution, download the :file:`*.tar.gz` file and look in the
:file:`examples` subdirectory.

To run the test suite:

* extract the :file:`lib/matplotlib/tests` or :file:`lib/mpl_toolkits/tests`
  directories from the source distribution;
* install test dependencies: `pytest <https://pypi.python.org/pypi/pytest>`_,
  Pillow, MiKTeX, GhostScript, ffmpeg, avconv, ImageMagick, and `Inkscape
  <https://inkscape.org/>`_;
* run ``python -mpytest``.

Third-party distributions of Matplotlib
=======================================

.. _install_scipy_dists:

Scientific Python Distributions
-------------------------------

`Anaconda <https://www.continuum.io/downloads/>`_ and `Canopy
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

* `Python <https://www.python.org/downloads/>`_ (>= 3.5)
* `FreeType <https://www.freetype.org/>`_ (>= 2.3)
* `libpng <http://www.libpng.org>`_ (>= 1.2)
* `NumPy <http://www.numpy.org>`_ (>= |minimum_numpy_version|)
* `setuptools <https://setuptools.readthedocs.io/en/latest/>`_
* `cycler <http://matplotlib.org/cycler/>`_ (>= 0.10.0)
* `dateutil <https://pypi.python.org/pypi/python-dateutil>`_ (>= 2.1)
* `kiwisolver <https://github.com/nucleic/kiwi>`_ (>= 1.0.0)
* `pyparsing <https://pyparsing.wikispaces.com/>`_

Optionally, you can also install a number of packages to enable better user
interface toolkits. See :ref:`what-is-a-backend` for more details on the
optional Matplotlib backends and the capabilities they provide.

* :term:`tk` (>= 8.3, != 8.6.0 or 8.6.1): for the Tk-based backends;
* `PyQt4 <https://pypi.python.org/pypi/PyQt4>`_ (>= 4.6) or
  `PySide <https://pypi.python.org/pypi/PySide>`_ (>= 1.0.3): for the Qt4-based
  backends;
* `PyQt5 <https://pypi.python.org/pypi/PyQt5>`_: for the Qt5-based backends;
* `PyGObject <https://pypi.org/project/PyGObject/>`_ or
  `pgi <https://pypi.org/project/pgi/>`_ (>= 0.0.11.2): for the GTK3-based
  backends;
* :term:`wxpython` (>= 4): for the WX-based backends;
* `cairocffi <https://cairocffi.readthedocs.io/en/latest/>`_ (>= 0.8) or
  `pycairo <https://pypi.python.org/pypi/pycairo>`_: for the cairo-based
  backends;
* `Tornado <https://pypi.python.org/pypi/tornado>`_: for the WebAgg backend;

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

   Matplotlib depends on non-Python libraries. `pkg-config
   <https://www.freedesktop.org/wiki/Software/pkg-config/>`_ can be used
   to find required non-Python libraries and thus make the install go more
   smoothly if the libraries and headers are not in the expected locations.

.. note::

  The following libraries are shipped with Matplotlib:

  - `Agg`: the Anti-Grain Geometry C++ rendering engine;
  - `qhull`: to compute Delaunay triangulation;
  - `ttconv`: a TrueType font utility.

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
https://github.com/conda-forge/python-feedstock.
