.. _dependencies:

============
Dependencies
============

Runtime dependencies
====================

Mandatory dependencies
----------------------

When installing through a package manager like ``pip`` or ``conda``, the
mandatory dependencies are automatically installed. This list is mainly for
reference.

* `Python <https://www.python.org/downloads/>`_ (>= 3.7)
* `NumPy <https://numpy.org>`_ (>= 1.17)
* `setuptools <https://setuptools.readthedocs.io/en/latest/>`_
* `cycler <https://matplotlib.org/cycler/>`_ (>= 0.10.0)
* `dateutil <https://pypi.org/project/python-dateutil/>`_ (>= 2.7)
* `kiwisolver <https://github.com/nucleic/kiwi>`_ (>= 1.0.1)
* `Pillow <https://pillow.readthedocs.io/en/latest/>`_ (>= 6.2)
* `pyparsing <https://pypi.org/project/pyparsing/>`_ (>=2.2.1)
* `fontTools <https://fonttools.readthedocs.io/en/latest/>`_ (>=4.22.0)


.. _optional_dependencies:

Optional dependencies
---------------------

The following packages and tools are not required but extend the capabilities
of Matplotlib.

Backends
~~~~~~~~

Matplotlib figures can be rendered to various user interfaces. See
:ref:`what-is-a-backend` for more details on the optional Matplotlib backends
and the capabilities they provide.

* Tk_ (>= 8.4, != 8.6.0 or 8.6.1) [#]_: for the Tk-based backends.
* PyQt6_ (>= 6.1), PySide6_, PyQt5_, or PySide2_: for the Qt-based backends.
* PyGObject_: for the GTK-based backends [#]_.
* wxPython_ (>= 4) [#]_: for the wx-based backends.
* pycairo_ (>= 1.11.0) or cairocffi_ (>= 0.8): for the GTK and/or cairo-based
  backends.
* Tornado_ (>=5): for the WebAgg backend.

.. _Tk: https://docs.python.org/3/library/tk.html
.. _PyQt5: https://pypi.org/project/PyQt5/
.. _PySide2: https://pypi.org/project/PySide2/
.. _PyQt6: https://pypi.org/project/PyQt6/
.. _PySide6: https://pypi.org/project/PySide6/
.. _PyGObject: https://pygobject.readthedocs.io/en/latest/
.. _wxPython: https://www.wxpython.org/
.. _pycairo: https://pycairo.readthedocs.io/en/latest/
.. _cairocffi: https://cairocffi.readthedocs.io/en/latest/
.. _Tornado: https://pypi.org/project/tornado/

.. [#] Tk is part of most standard Python installations, but it's not part of
       Python itself and thus may not be present in rare cases.
.. [#] If using pip (and not conda), PyGObject must be built from source; see
       https://pygobject.readthedocs.io/en/latest/devguide/dev_environ.html.
.. [#] If using pip (and not conda) on Linux, wxPython wheels must be manually
       downloaded from https://wxpython.org/pages/downloads/.

Animations
~~~~~~~~~~

* `ffmpeg <https://www.ffmpeg.org/>`_: for saving movies.
* `ImageMagick <https://www.imagemagick.org/script/index.php>`_: for saving
  animated gifs.

Font handling and rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `LaTeX <https://www.latex-project.org/>`_ (with `cm-super
  <https://ctan.org/pkg/cm-super>`__ ) and `GhostScript (>=9.0)
  <https://ghostscript.com/download/>`_ : for rendering text with LaTeX.
* `fontconfig <https://www.fontconfig.org>`_ (>= 2.7): for detection of system
  fonts on Linux.

C libraries
-----------

Matplotlib brings its own copies of the following libraries:

- ``Agg``: the Anti-Grain Geometry C++ rendering engine
- ``ttconv``: a TrueType font utility

Additionally, Matplotlib depends on:

- FreeType_ (>= 2.3): a font rendering library
- QHull_ (>= 2020.2): a library for computing triangulations

.. _FreeType: https://www.freetype.org/
.. _Qhull: http://www.qhull.org/

By default, Matplotlib downloads and builds its own copies of FreeType (this is
necessary to run the test suite, because different versions of FreeType
rasterize characters differently) and of Qhull.  As an exception, Matplotlib
defaults to the system version of FreeType on AIX.

To force Matplotlib to use a copy of FreeType or Qhull already installed in
your system, create a :file:`mplsetup.cfg` file with the following contents:

.. code-block:: cfg

   [libs]
   system_freetype = true
   system_qhull = true

before running ``python -m pip install .``.

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
standard environment variables -- on Linux and OSX:

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


.. _development-dependencies:

Additional dependencies for development
=======================================

.. _test-dependencies:

Additional dependencies for testing
===================================
This section lists the additional software required for
:ref:`running the tests <testing>`.

Required:

- pytest_ (>=3.6)
- Ghostscript_ (>= 9.0, to render PDF files)
- Inkscape_ (to render SVG files)

Optional:

- pytest-cov_ (>=2.3.1) to collect coverage information
- pytest-flake8_ to test coding standards using flake8_
- pytest-timeout_ to limit runtime in case of stuck tests
- pytest-xdist_ to run tests in parallel

.. _pytest: http://doc.pytest.org/en/latest/
.. _Ghostscript: https://www.ghostscript.com/
.. _Inkscape: https://inkscape.org
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _pytest-flake8: https://pypi.org/project/pytest-flake8/
.. _pytest-xdist: https://pypi.org/project/pytest-xdist/
.. _pytest-timeout: https://pypi.org/project/pytest-timeout/
.. _flake8: https://pypi.org/project/flake8/


.. _doc-dependencies:

Additional dependencies for building documentation
==================================================

Python packages
---------------
The additional Python packages required to build the
:ref:`documentation <documenting-matplotlib>` are listed in
:file:`doc-requirements.txt` and can be installed using ::

    pip install -r requirements/doc/doc-requirements.txt

The content of :file:`doc-requirements.txt` is also shown below:

   .. include:: ../../requirements/doc/doc-requirements.txt
      :literal:

Additional external dependencies
--------------------------------
Required:

* a minimal working LaTeX distribution
* `Graphviz <http://www.graphviz.org/download>`_
* the following LaTeX packages (if your OS bundles TeXLive, the
  "complete" version of the installer, e.g. "texlive-full" or "texlive-all",
  will often automatically include these packages):

  * `cm-super <https://ctan.org/pkg/cm-super>`_
  * `dvipng <https://ctan.org/pkg/dvipng>`_
  * `underscore <https://ctan.org/pkg/underscore>`_

Optional, but recommended:

* `Inkscape <https://inkscape.org>`_
* `optipng <http://optipng.sourceforge.net>`_
* the font "Humor Sans" (aka the "XKCD" font), or the free alternative
  `Comic Neue <http://comicneue.com/>`_
* the font "Times New Roman"

.. note::

  The documentation will not build without LaTeX and Graphviz.  These are not
  Python packages and must be installed separately. The documentation can be
  built without Inkscape and optipng, but the build process will raise various
  warnings. If the build process warns that you are missing fonts, make sure
  your LaTeX distribution bundles cm-super or install it separately.
