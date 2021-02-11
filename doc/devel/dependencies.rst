.. _dependencies:

============
Dependencies
============

Mandatory dependencies
======================

When installing through a package manager like ``pip`` or ``conda``, the
mandatory dependencies are automatically installed. This list is mainly for
reference.

* `Python <https://www.python.org/downloads/>`_ (>= 3.7)
* `NumPy <https://numpy.org>`_ (>= 1.16)
* `setuptools <https://setuptools.readthedocs.io/en/latest/>`_
* `cycler <https://matplotlib.org/cycler/>`_ (>= 0.10.0)
* `dateutil <https://pypi.org/project/python-dateutil>`_ (>= 2.7)
* `kiwisolver <https://github.com/nucleic/kiwi>`_ (>= 1.0.1)
* `Pillow <https://pillow.readthedocs.io/en/latest/>`_ (>= 6.2)
* `pyparsing <https://pypi.org/project/pyparsing/>`_ (>=2.2.1)


.. _optional_dependencies:

Optional dependencies
=====================

The following packages and tools are not required but extend the capabilities
of Matplotlib.

Backends
--------

Matplotlib figures can be rendered to various user interfaces. See
:ref:`what-is-a-backend` for more details on the optional Matplotlib backends
and the capabilities they provide.

* Tk_ (>= 8.3, != 8.6.0 or 8.6.1) [#]_: for the Tk-based backends.
* PyQt4_ (>= 4.6) or PySide_ (>= 1.0.3) [#]_: for the Qt4-based backends.
* PyQt5_ or PySide2_: for the Qt5-based backends.
* PyGObject_: for the GTK3-based backends [#]_.
* wxPython_ (>= 4) [#]_: for the wx-based backends.
* pycairo_ (>= 1.11.0) or cairocffi_ (>= 0.8): for the GTK3 and/or cairo-based
  backends.
* Tornado_: for the WebAgg backend.

.. _Tk: https://docs.python.org/3/library/tk.html
.. _PyQt4: https://pypi.org/project/PyQt4
.. _PySide: https://pypi.org/project/PySide
.. _PyQt5: https://pypi.org/project/PyQt5
.. _PySide2: https://pypi.org/project/PySide2
.. _PyGObject: https://pygobject.readthedocs.io/en/latest/
.. _wxPython: https://www.wxpython.org/
.. _pycairo: https://pycairo.readthedocs.io/en/latest/
.. _cairocffi: https://cairocffi.readthedocs.io/en/latest/
.. _Tornado: https://pypi.org/project/tornado

.. [#] Tk is part of most standard Python installations, but it's not part of
       Python itself and thus may not be present in rare cases.
.. [#] PySide cannot be pip-installed on Linux (but can be conda-installed).
.. [#] If using pip (and not conda), PyGObject must be built from source; see
       https://pygobject.readthedocs.io/en/latest/devguide/dev_environ.html.
.. [#] If using pip (and not conda) on Linux, wxPython wheels must be manually
       downloaded from https://wxpython.org/pages/downloads/.

Animations
----------

* `ffmpeg <https://www.ffmpeg.org/>`_: for saving movies.
* `ImageMagick <https://www.imagemagick.org/script/index.php>`_: for saving
  animated gifs.

Font handling and rendering
---------------------------

* `LaTeX <https://www.latex-project.org/>`_ (with `cm-super
  <https://ctan.org/pkg/cm-super>`__ ) and `GhostScript (>=9.0)
  <https://ghostscript.com/download/>`_ : for rendering text with LaTeX.
* `fontconfig <https://www.fontconfig.org>`_ (>= 2.7): for detection of system
  fonts on Linux.

C libraries
===========

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
your system, create a :file:`setup.cfg` file with the following contents:

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
