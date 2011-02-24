.. _installing:

**********
Installing
**********

There are lots of different ways to install matplotlib, and the best
way depends on what operating system you are using, what you already
have installed, and how you want to use it.  To avoid wading through
all the details (and potential complications) on this page, the
easiest thing for you to do is use one of the pre-packaged python
distributions that already provide matplotlib built in.  The Enthought
Python Distribution `(EPD)
<http://www.enthought.com/products/epd.php>`_ for Windows, OS X or
Redhat is an excellent choice that "just works" out of the box.
Another excellent alternative for Windows users is `Python (x, y)
<http://www.pythonxy.com/foreword.php>`_ which tends to be updated a
bit more frequently.  Both of these packages include matplotlib and
pylab, and *lots* of other useful tools.  matplotlib is also packaged
for pretty much every major linux distribution, so if you are on linux
your package manager will probably provide matplotlib prebuilt.

One single click installer and you are done.

OK, so you want to do it the hard way?
======================================

For some people, the prepackaged pythons discussed above are not an
option.  That's OK, it's usually pretty easy to get a custom install
working.  You will first need to find out if you have python installed
on your machine, and if not, install it.  The official python builds
are available for download `here <http://www.python.org/download>`_,
but OS X users please read :ref:`which-python-for-osx`.

Once you have python up and running, you will need to install `numpy
<http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103>`_.
numpy provides high performance array data structures and mathematical
functions, and is a requirement for matplotlib.  You can test your
progress::

    >>> import numpy
    >>> print numpy.__version__

matplotlib requires numpy version 1.1 or later.  Although it is not a
requirement to use matplotlib, we strongly encourage you to install
`ipython <http://ipython.scipy.org/dist>`_, which is an interactive
shell for python that is matplotlib aware.

Next we need to get matplotlib installed.  We provide prebuilt
binaries for OS X and Windows on the matplotlib `download
<http://sourceforge.net/projects/matplotlib/files/>`_ page.  Click on
the latest release of the "matplotlib" package, choose your python
version (2.5 or 2.6) and your platform (macosx or win32) and you
should be good to go.  If you have any problems, please check the
:ref:`installing-faq`, google around a little bit, and post a question
the `mailing list
<http://sourceforge.net/project/showfiles.php?group_id=80706>`_.  If
you are on debian/unbuntu linux, it suffices to do::

  > sudo apt-get install python-matplotlib

Instructions for installing our OSX binaries are found in the FAQ
:ref:`install_osx_binaries`.


Once you have ipython, numpy and matplotlib installed, in ipython's
"pylab" mode you have a MATLAB-like environment that automatically
handles most of the configuration details for you, so you can get up
and running quickly::

    johnh@flag:~> ipython -pylab
    Python 2.4.5 (#4, Apr 12 2008, 09:09:16)
    IPython 0.9.0 -- An enhanced Interactive Python.

      Welcome to pylab, a matplotlib-based Python environment.
      For more information, type 'help(pylab)'.

    In [1]: x = randn(10000)

    In [2]: hist(x, 100)

Instructions for installing our OSX binaries are found in the FAQ
ref:`install_osx_binaries`.

Note that when testing matplotlib installations from the interactive
python console, there are some issues relating to user interface
toolkits and interactive settings that are discussed in
:ref:`mpl-shell`.

.. _install_from_source:

Installing from source
======================

If you are interested perhaps in contributing to matplotlib
development, running the latest greatest code, or just like to
build everything yourself, it is not difficult to build matplotlib
from source.  Grab the latest *tar.gz* release file from `sourceforge
<http://sourceforge.net/project/showfiles.php?group_id=80706>`_, or if
you want to develop matplotlib or just need the latest bugfixed
version, grab the latest git version :ref:`install-from-git`.

Once you have satisfied the requirements detailed below (mainly
python, numpy, libpng and freetype), you build matplotlib in the usual
way::

  cd matplotlib
  python setup.py build
  python setup.py install

We provide a `setup.cfg
<https://github.com/matplotlib/matplotlib/raw/master/setup.cfg.template>`_
file that lives along :file:`setup.py` which you can use to customize
the build process, for example, which default backend to use, whether
some of the optional libraries that matplotlib ships with are
installed, and so on.  This file will be particularly useful to those
packaging matplotlib.


.. _install_requirements:

Build requirements
==================

These are external packages which you will need to install before
installing matplotlib. Windows users only need the first two (python
and numpy) since the others are built into the matplotlib windows
installers available for download at the sourceforge site.  If you are
building on OSX, see :ref:`build_osx`. If you are installing
dependencies with a package manager, you may need to install the
development packages (look for a "-dev" postfix) in addition to the
libraries themselves.

:term:`python` 2.4 (or later but not python3)
    matplotlib requires python 2.4 or later (`download <http://www.python.org/download/>`__)

:term:`numpy` 1.1 (or later)
    array support for python (`download
    <http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103>`__)

libpng 1.1 (or later)
    library for loading and saving :term:`PNG` files (`download
    <http://www.libpng.org/pub/png/libpng.html>`__). libpng requires
    zlib. If you are a windows user, you can ignore this since we
    build support into the matplotlib single click installer

:term:`freetype` 1.4 (or later)
    library for reading true type font files. If you are a windows
    user, you can ignore this since we build support into the
    matplotlib single click installer.

**Optional**

These are optional packages which you may want to install to use
matplotlib with a user interface toolkit. See
:ref:`what-is-a-backend` for more details on the optional matplotlib
backends and the capabilities they provide

:term:`tk` 8.3 or later
    The TCL/Tk widgets library used by the TkAgg backend

:term:`pyqt` 3.1 or later
    The Qt3 widgets library python wrappers for the QtAgg backend

:term:`pyqt` 4.0 or later
    The Qt4 widgets library python wrappers for the Qt4Agg backend

:term:`pygtk` 2.4 or later
    The python wrappers for the GTK widgets library for use with the
    GTK or GTKAgg backend

:term:`wxpython` 2.6 or later
    The python wrappers for the wx widgets library for use with the
    WXAgg backend

:term:`wxpython` 2.8 or later
    The python wrappers for the wx widgets library for use with the
    WX backend

:term:`pyfltk` 1.0 or later
    The python wrappers of the FLTK widgets library for use with FLTKAgg

**Required libraries that ship with matplotlib**

:term:`agg` 2.4
    The antigrain C++ rendering engine.  matplotlib links against the
    agg template source statically, so it will not affect anything on
    your system outside of matplotlib.

:term:`pytz` 2007g or later
    timezone handling for python datetime objects.  By default,
    matplotlib will install pytz if it isn't already installed on your
    system.  To override the default, use :file:`setup.cfg` to force or
    prevent installation of pytz.

:term:`dateutil` 1.1 or later
    provides extensions to python datetime handling.  By default, matplotlib
    will install dateutil if it isn't already installed on your
    system.  To override the default, use :file:`setup.cfg` to force
    or prevent installation of dateutil.



.. _build_osx:

Building on OSX
==================

The build situation on OSX is complicated by the various places one
can get the png and freetype requirements from (darwinports, fink,
/usr/X11R6) and the different architectures (x86, ppc, universal) and
the different OSX version (10.4 and 10.5). We recommend that you build
the way we do for the OSX release: by grabbing the tarbar or git
repository, cd-ing into the release/osx dir, and following the
instruction in the README.  This directory has a Makefile which will
automatically grab the zlib, png and freetype dependencies from the
web, build them with the right flags to make universal libraries, and
then build the matplotlib source and binary installers.

