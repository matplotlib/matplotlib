.. redirect-from:: /devel/dependencies
.. redirect-from:: /users/installing/dependencies

.. _dependencies:

************
Dependencies
************

.. _runtime_dependencies:

Runtime dependencies
====================


Required
--------

When installing through a package manager like ``pip`` or ``conda``, the
mandatory dependencies are automatically installed. This list is mainly for
reference.

* `Python <https://www.python.org/downloads/>`_ (>= 3.9)
* `contourpy <https://pypi.org/project/contourpy/>`_ (>= 1.0.1)
* `cycler <https://matplotlib.org/cycler/>`_ (>= 0.10.0)
* `dateutil <https://pypi.org/project/python-dateutil/>`_ (>= 2.7)
* `fontTools <https://fonttools.readthedocs.io/en/latest/>`_ (>= 4.22.0)
* `kiwisolver <https://github.com/nucleic/kiwi>`_ (>= 1.3.1)
* `NumPy <https://numpy.org>`_ (>= 1.23)
* `packaging <https://pypi.org/project/packaging/>`_ (>= 20.0)
* `Pillow <https://pillow.readthedocs.io/en/latest/>`_ (>= 8.0)
* `pyparsing <https://pypi.org/project/pyparsing/>`_ (>= 2.3.1)
* `importlib-resources <https://pypi.org/project/importlib-resources/>`_
  (>= 3.2.0; only required on Python < 3.10)


.. _optional_dependencies:

Optional
--------

The following packages and tools are not required but extend the capabilities
of Matplotlib.

.. _backend_dependencies:

Backends
^^^^^^^^

Matplotlib figures can be rendered to various user interfaces. See
:ref:`what-is-a-backend` for more details on the optional Matplotlib backends
and the capabilities they provide.

* Tk_ (>= 8.5, != 8.6.0 or 8.6.1): for the Tk-based backends. Tk is part of
  most standard Python installations, but it's not part of Python itself and
  thus may not be present in rare cases.
* PyQt6_ (>= 6.1), PySide6_, PyQt5_ (>= 5.12), or PySide2_: for the Qt-based
  backends.
* PyGObject_ and pycairo_ (>= 1.14.0): for the GTK-based backends. If using pip
  (but not conda or system package manager) PyGObject must be built from
  source; see `pygobject documentation
  <https://pygobject.readthedocs.io/en/latest/devguide/dev_environ.html>`_.
* pycairo_ (>= 1.14.0) or cairocffi_ (>= 0.8): for cairo-based backends.
* wxPython_ (>= 4): for the wx-based backends.  If using pip (but not conda or
  system package manager) on Linux wxPython wheels must be manually downloaded
  from https://wxpython.org/pages/downloads/.
* Tornado_ (>= 5): for the WebAgg backend.
* ipykernel_: for the nbagg backend.
* macOS (>= 10.12): for the macosx backend.

.. _Tk: https://docs.python.org/3/library/tk.html
.. _PyQt5: https://pypi.org/project/PyQt5/
.. _PySide2: https://pypi.org/project/PySide2/
.. _PyQt6: https://pypi.org/project/PyQt6/
.. _PySide6: https://pypi.org/project/PySide6/
.. _PyGObject: https://pygobject.readthedocs.io/en/latest/
.. _wxPython: https://www.wxpython.org/
.. _pycairo: https://pycairo.readthedocs.io/en/latest/
.. _cairocffi: https://doc.courtbouillon.org/cairocffi/stable/
.. _Tornado: https://pypi.org/project/tornado/
.. _ipykernel: https://pypi.org/project/ipykernel/

Animations
^^^^^^^^^^

* `ffmpeg <https://www.ffmpeg.org/>`_: for saving movies.
* `ImageMagick <https://www.imagemagick.org/script/index.php>`_: for saving
  animated gifs.

Font handling and rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `LaTeX <https://www.latex-project.org/>`_ (with `cm-super
  <https://ctan.org/pkg/cm-super>`__ and `underscore
  <https://ctan.org/pkg/underscore>`__) and `GhostScript (>= 9.0)
  <https://ghostscript.com/releases/>`_: for rendering text with LaTeX.
* `fontconfig <https://www.fontconfig.org>`_ (>= 2.7): for detection of system
  fonts on Linux.

C libraries
-----------

Matplotlib brings its own copies of the following libraries:

- ``Agg``: the Anti-Grain Geometry C++ rendering engine
- ``ttconv``: a TrueType font utility

Additionally, Matplotlib depends on:

- FreeType_ (>= 2.3): a font rendering library
- QHull_ (>= 8.0.2): a library for computing triangulations (note that this version is
  also known as 2020.2)

.. _FreeType: https://www.freetype.org/
.. _Qhull: http://www.qhull.org/


Download during install
^^^^^^^^^^^^^^^^^^^^^^^

By default, Matplotlib downloads and builds its own copies of Qhull and FreeType.
The vendored version of FreeType is necessary to run the test suite, because
different versions of FreeType rasterize characters differently.


Use system libraries
^^^^^^^^^^^^^^^^^^^^

To force Matplotlib to use a copy of FreeType or Qhull already installed in your system,
you must `pass configuration settings to Meson via meson-python
<https://meson-python.readthedocs.io/en/stable/how-to-guides/config-settings.html>`_:

.. code-block:: sh

   python -m pip install \
     --config-settings=setup-args="-Dsystem-freetype=true" \
     --config-settings=setup-args="-Dsystem-qhull=true" \
     .


In this case, you need to install the FreeType and Qhull library and headers.
This can be achieved using a package manager, e.g. for FreeType:

.. code-block:: sh

   # Pick ONE of the following:
   sudo apt install libfreetype6-dev  # Debian/Ubuntu
   sudo dnf install freetype-devel    # Fedora
   brew install freetype              # macOS with Homebrew
   conda install freetype             # conda, any OS

(adapt accordingly for Qhull).

On Linux and macOS, it is also recommended to install pkg-config_, a helper
tool for locating FreeType:

.. code-block:: sh

   # Pick ONE of the following:
   sudo apt install pkg-config  # Debian/Ubuntu
   sudo dnf install pkgconf     # Fedora
   brew install pkg-config      # macOS with Homebrew
   conda install pkg-config     # conda
   # Or point the PKG_CONFIG environment variable to the path to pkg-config:
   export PKG_CONFIG=...

.. _pkg-config: https://www.freedesktop.org/wiki/Software/pkg-config/

If not using pkg-config (in particular on Windows), you may need to set the
include path (to the library headers) and link path (to the libraries)
explicitly, if they are not in standard locations.  This can be done using
standard environment variables -- on Linux and macOS:

.. code-block:: sh

   export CFLAGS='-I/directory/containing/ft2build.h'
   export LDFLAGS='-L/directory/containing/libfreetype.so'

and on Windows:

.. code-block:: bat

   set CL=/IC:\directory\containing\ft2build.h
   set LINK=/LIBPATH:C:\directory\containing\freetype.lib

If you go this route but need to reset and rebuild to change your settings,
remember to clear your artifacts before re-building::

  git clean -xfd

From source files
^^^^^^^^^^^^^^^^^

If the automatic download does not work (for example, on air-gapped systems) it is
preferable to instead use system libraries. However you can manually download the
tarballs into :file:`subprojects/packagecache` at the top level of the checkout
repository. The expected SHA256 hashes of the downloaded tarballs are in
:file:`subprojects/*.wrap` if you wish to verify them, but they will also be checked by
the build system before unpacking.


Minimum pip / manylinux support (linux)
---------------------------------------

Matplotlib publishes `manylinux wheels <https://github.com/pypa/manylinux>`_
which have a minimum version of pip which will recognize the wheels

- Python 3.9+: ``manylinux2014`` / pip >= 19.3

In all cases the required version of pip is embedded in the CPython source.



.. _development-dependencies:

Build dependencies
==================


.. _setup-dependencies:

Python
------

By default, ``pip`` will build packages using build isolation, meaning that these
build dependencies are temporally installed by pip for the duration of the
Matplotlib build process. However, build isolation is disabled when :ref:`installing Matplotlib for development <development-install>`;
therefore we recommend using one of our :ref:`virtual environment configurations <dev-environment>` to
create a development environment in which these packages are automatically installed.

If you are developing Matplotlib and unable to use our environment configurations,
then you must manually install the following packages into your development environment:

- `meson-python <https://meson-python.readthedocs.io/>`_ (>= 0.13.1).
- `ninja <https://ninja-build.org/>`_ (>= 1.8.2). This may be available in your package
  manager or bundled with Meson, but may be installed via ``pip`` if otherwise not
  available.
- `PyBind11 <https://pypi.org/project/pybind11/>`_ (>= 2.6). Used to connect C/C++ code
  with Python.
- `setuptools_scm <https://pypi.org/project/setuptools-scm/>`_ (>= 7).  Used to
  update the reported ``mpl.__version__`` based on the current git commit.
  Also a runtime dependency for editable installs.
- `NumPy <https://numpy.org>`_ (>= 1.22).  Also a runtime dependency.


.. _compile-dependencies:

Compiled extensions
-------------------

Matplotlib requires a C++ compiler that supports C++17, and each platform has a
development environment that must be installed before a compiler can be installed.
You may also need to install headers for various libraries used in the compiled extension
source files.

.. tab-set::

    .. tab-item:: Linux

        On some Linux systems, you can install a meta-build package. For example,
        on  Ubuntu ``apt install build-essential``

        Otherwise, use the system distribution's package manager to install
        :ref:`gcc <compiler-table>`.

    .. tab-item:: macOS

        Install `Xcode <https://developer.apple.com/xcode/>`_ for Apple platform development.

    .. tab-item:: Windows

        Install `Visual Studio Build Tools <https://visualstudio.microsoft.com/downloads/?q=build+tools>`_

        Make sure "Desktop development with C++" is selected, and that the latest MSVC,
        "C++ CMake tools for Windows," and a Windows SDK compatible with your version
        of Windows are selected and installed. They should be selected by default under
        the "Optional" subheading, but are required to build Matplotlib from source.

        Alternatively, you can install a Linux-like environment such as `CygWin <https://www.cygwin.com/>`_
        or `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_.
        If using `MinGW-64 <https://www.mingw-w64.org/>`_, we require **v6** of the
        ```Mingw-w64-x86_64-headers``.


We highly recommend that you install a compiler using your platform tool, i.e.,
Xcode, VS Code or Linux package manager. Choose **one** compiler from this list:

.. _compiler-table:

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - compiler
     - minimum version
     - platforms
     - notes
   * - GCC
     - **7.2**
     - Linux, macOS, Windows
     - `gcc 7.2 <https://gcc.gnu.org/projects/cxx-status.html#cxx17>`_,
       `GCC: Binaries <https://gcc.gnu.org/install/binaries.html>`_,
   * - Clang (LLVM)
     - **5**
     - Linux, macOS
     - `clang 5 <https://clang.llvm.org/cxx_status.html>`_, `LLVM <https://releases.llvm.org/download.html>`_
   * - MSVC++
     - **16.0**
     - Windows
     - `Visual Studio 2019 C++ <https://docs.microsoft.com/en-us/cpp/overview/visual-cpp-language-conformance?view=msvc-160>`_



.. _test-dependencies:

Test dependencies
=================

This section lists the additional software required for
:ref:`running the tests <testing>`.

Required
--------

- pytest_ (>= 7.0.0)

Optional
--------

In addition to all of the optional dependencies on the main library, for
testing the following will be used if they are installed.

- Ghostscript_ (>= 9.0, to render PDF files)
- Inkscape_ (to render SVG files)
- nbformat_ and nbconvert_ used to test the notebook backend
- pandas_ used to test compatibility with Pandas
- pikepdf_ used in some tests for the pgf and pdf backends
- psutil_ used in testing the interactive backends
- pytest-cov_ (>= 2.3.1) to collect coverage information
- pytest-flake8_ to test coding standards using flake8_
- pytest-timeout_ to limit runtime in case of stuck tests
- pytest-xdist_ to run tests in parallel
- pytest-xvfb_ to run tests without windows popping up (Linux)
- pytz_ used to test pytz int
- sphinx_ used to test our sphinx extensions
- `WenQuanYi Zen Hei`_ and `Noto Sans CJK`_ fonts for testing font fallback and
  non-Western fonts
- xarray_ used to test compatibility with xarray

If any of these dependencies are not discovered, then the tests that rely on
them will be skipped by pytest.

.. note::

  When installing Inkscape on Windows, make sure that you select “Add
  Inkscape to system PATH”, either for all users or current user, or the
  tests will not find it.

.. _Ghostscript: https://ghostscript.com/
.. _Inkscape: https://inkscape.org
.. _flake8: https://pypi.org/project/flake8/
.. _nbconvert: https://pypi.org/project/nbconvert/
.. _nbformat: https://pypi.org/project/nbformat/
.. _pandas: https://pypi.org/project/pandas/
.. _pikepdf: https://pypi.org/project/pikepdf/
.. _psutil: https://pypi.org/project/psutil/
.. _pytz: https://fonts.google.com/noto/use#faq
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _pytest-flake8: https://pypi.org/project/pytest-flake8/
.. _pytest-timeout: https://pypi.org/project/pytest-timeout/
.. _pytest-xdist: https://pypi.org/project/pytest-xdist/
.. _pytest-xvfb: https://pypi.org/project/pytest-xvfb/
.. _pytest: http://doc.pytest.org/en/latest/
.. _sphinx: https://pypi.org/project/Sphinx/
.. _WenQuanYi Zen Hei: http://wenq.org/en/
.. _Noto Sans CJK: https://fonts.google.com/noto/use
.. _xarray: https://pypi.org/project/xarray/


.. _doc-dependencies:

Documentation dependencies
==========================

Python
------

The additional Python packages required to build the
:ref:`documentation <documenting-matplotlib>` are listed in
:file:`doc-requirements.txt` and can be installed using ::

    pip install -r requirements/doc/doc-requirements.txt

The content of :file:`doc-requirements.txt` is also shown below:

.. include:: ../../requirements/doc/doc-requirements.txt
   :literal:


External tools
--------------

The documentation requires LaTeX and Graphviz.  These are not
Python packages and must be installed separately.

Required
^^^^^^^^

* `Graphviz <http://www.graphviz.org/download>`_
* a minimal working LaTeX distribution, e.g. `TeX Live <https://www.tug.org/texlive/>`_ or
  `MikTeX <https://miktex.org/>`_

The following LaTeX packages:

  * `dvipng <https://ctan.org/pkg/dvipng>`_
  * `underscore <https://ctan.org/pkg/underscore>`_
  * `cm-super <https://ctan.org/pkg/cm-super>`_
  * ``collection-fontsrecommended``

The complete version of many LaTex distribution installers, e.g.
"texlive-full" or "texlive-all", will often automatically include these packages.


Optional
^^^^^^^^

The documentation can be built without Inkscape and optipng, but the build
process will raise various warnings.

* `Inkscape <https://inkscape.org>`_
* `optipng <http://optipng.sourceforge.net>`_
* the font `xkcd script <https://github.com/ipython/xkcd-font/>`_ or `Comic Neue <http://comicneue.com/>`_
* the font "Times New Roman"
