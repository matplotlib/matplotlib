.. _installing-faq:

*************
 Installation
*************

.. contents::
   :backlinks: none

Report a compilation problem
============================

See :ref:`reporting-problems`.

matplotlib compiled fine, but nothing shows up when I use it
============================================================

The first thing to try is a :ref:`clean install <clean-install>` and see if
that helps.  If not, the best way to test your install is by running a script,
rather than working interactively from a python shell or an integrated
development environment such as :program:`IDLE` which add additional
complexities. Open up a UNIX shell or a DOS command prompt and cd into a
directory containing a minimal example in a file. Something like
:file:`simple_plot.py` for example::

  from pylab import *
  plot([1,2,3])
  show()

and run it with::

  python simple_plot.py --verbose-helpful

This will give you additional information about which backends matplotlib is
loading, version information, and more. At this point you might want to make
sure you understand matplotlib's :ref:`configuration <customizing-matplotlib>`
process, governed by the :file:`matplotlibrc` configuration file which contains
instructions within and the concept of the matplotlib backend.

If you are still having trouble, see :ref:`reporting-problems`.

.. _clean-install:

How to completely remove matplotlib
===================================

Occasionally, problems with matplotlib can be solved with a clean
installation of the package.

The process for removing an installation of matplotlib depends on how
matplotlib was originally installed on your system. Follow the steps
below that goes with your original installation method to cleanly
remove matplotlib from your system.

Easy Install
------------

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Run::

     easy_install -m matplotlib


3. Delete any .egg files or directories from your :ref:`installation
   directory <locating-matplotlib-install>`.



Windows installer
-----------------

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Use :menuselection:`Start --> Control Panel` to start the :program:`Add and
   Remove Software` utility.

Source install
--------------

Unfortunately::

    python setup.py clean

does not properly clean the build directory, and does nothing to the
install directory.  To cleanly rebuild:

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Delete the ``build`` directory in the source tree.

3. Delete any matplotlib directories or eggs from your :ref:`installation
   directory <locating-matplotlib-install>`.

How to Install
==============

.. _install-from-git:

Source install from git
-----------------------

Clone the main source using one of::

   git clone git@github.com:matplotlib/matplotlib.git

or::

   git clone git://github.com/matplotlib/matplotlib.git

and build and install as usual with::

  > cd matplotlib
  > python setup.py install

.. note::

    If you are on debian/ubuntu, you can get all the dependencies
    required to build matplotlib with::

      sudo apt-get build-dep python-matplotlib

    If you are on Fedora/RedHat, you can get all the dependencies
    required to build matplotlib by first installing ``yum-builddep``
    and then running::

       su -c "yum-builddep python-matplotlib"

    This does not build matplotlib, but it does get all of the
    build dependencies, which will make building from source easier.


If you want to be able to follow the development branch as it changes
just replace the last step with (make sure you have **setuptools**
installed)::

  > python setup.py develop

This creates links in the right places and installs the command
line script to the appropriate places.

.. note::
   Mac OSX users please see the :ref:`build_osx` guide.

   Windows users please see the :ref:`build_windows` guide.

Then, if you want to update your matplotlib at any time, just do::

  > git pull

When you run `git pull`, if the output shows that only Python files have been
updated, you are all set. If C files have changed, you need to run the `python
setup.py develop` command again to compile them.

There is more information on :ref:`using git <using-git>` in
the developer docs.


Linux Notes
===========

Because most Linux distributions use some sort of package manager,
we do not provide a pre-built binary for the Linux platform.
Instead, we recommend that you use the "Add Software" method for
your system to install matplotlib. This will guarantee that everything
that is needed for matplotlib will be installed as well.

If, for some reason, you can not use the package manager, Linux usually
comes with at least a basic build system. Follow the :ref:`instructions
<install-from-git>` found above for how to build and install matplotlib.


OS-X Notes
==========

.. _which-python-for-osx:

Which python for OS X?
----------------------

Apple ships OS X with its own Python, in ``/usr/bin/python``, and its own copy
of matplotlib. Unfortunately, the way Apple currently installs its own copies
of numpy, scipy and matplotlib means that these packages are difficult to
upgrade (see `system python packages`_).  For that reason we strongly suggest
that you install a fresh version of Python and use that as the basis for
installing libraries such as numpy and matplotlib.  One convenient way to
install matplotlib with other useful Python software is to use one of the
excellent Python scientific software collections that are now available:

.. _system python packages:
    https://github.com/MacPython/wiki/wiki/Which-Python#system-python-and-extra-python-packages

- Anaconda_ from `Continuum Analytics`_
- Canopy_ from Enthought_

.. _Canopy: https://enthought.com/products/canopy/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Enthought: http://enthought.com
.. _Continuum Analytics: http://continuum.io

These collections include Python itself and a wide range of libraries; if you
need a library that is not available from the collection, you can install it
yourself using standard methods such as *pip*.  Continuum and Enthought offer
their own installation support for these collections; see the Ananconda and
Canopy web pages for more information.

Other options for a fresh Python install are the standard installer from
`python.org <https://www.python.org/downloads/mac-osx/>`_, or installing
Python using a general OSX package management system such as `homebrew
<http://brew.sh>`_ or `macports <http://www.macports.org>`_.  Power users on
OSX will likely want one of homebrew or macports on their system to install
open source software packages, but it is perfectly possible to use these
systems with another source for your Python binary, such as Anaconda, Canopy
or Python.org Python.

.. _install_osx_binaries:

Installing OSX binary wheels
----------------------------

If you are using recent Python from http://www.python.org, Macports or
Homebrew, then you can use the standard pip installer to install matplotlib
binaries in the form of wheels.

Python.org Python
^^^^^^^^^^^^^^^^^

Install pip following the `standard pip install instructions
<http://pip.readthedocs.org/en/latest/installing.html>`_.  For the impatient,
open a new Terminal.app window and::

    curl -O https://bootstrap.pypa.io/get-pip.py

Then (Python 2.7)::

    python get-pip.py

or (Python 3)::

    python3 get-pip.py

You can now install matplotlib and all its dependencies with::

    pip install matplotlib

Macports
^^^^^^^^

For Python 2.7::

    sudo port install py27-pip
    sudo pip-2.7 install matplotlib

For Python 3.4::

    sudo port install py34-pip
    sudo pip-3.4 install matplotlib

Homebrew
^^^^^^^^

For Python 2.7::

    pip2 install matplotlib

For Python 3.4::

    pip3 install matplotlib

You might also want to install IPython; we recommend you install IPython with
the IPython notebook option, like this:

* Python.org Python:  ``pip install ipython[notebook]``
* Macports ``sudo pip-2.7 install ipython[notebook]`` or ``sudo pip-3.4
  install ipython[notebook]``
* Homebrew ``pip2 install ipython[notebook]`` or ``pip3 install
  ipython[notebook]``

Pip problems
^^^^^^^^^^^^

If you get errors with pip trying to run a compiler like ``gcc`` or ``clang``,
then the first thing to try is to `install xcode
<https://guide.macports.org/chunked/installing.html#installing.xcode>`_ and
retry the install.  If that does not work, then check
:ref:`reporting-problems`.

Installing via OSX mpkg installer package
-----------------------------------------

matplotlib also has a disk image (``.dmg``) installer, which contains a
typical Installer.app package to install matplotlib.  You should use binary
wheels instead of the disk image installer if you can, because:

* wheels work with Python.org Python, homebrew and macports, the disk image
  installer only works with Python.org Python.
* The disk image installer doesn't check for recent versions of packages that
  matplotlib depends on, and unconditionally installs the versions of
  dependencies contained in the disk image installer.  This can overwrite
  packages that you have already installed, which might cause problems for
  other packages, if you have a pre-existing Python.org setup on your
  computer.

If you still want to use the disk image installer, read on.

.. note::
   Before installing via the disk image installer, be sure that all of the
   packages were compiled for the same version of python.  Often, the download
   site for NumPy and matplotlib will display a supposed 'current' version of
   the package, but you may need to choose a different package from the full
   list that was built for your combination of python and OSX.

The disk image installer will have a ``.dmg`` extension, and will have a name
like :file:`matplotlib-1.4.0-py2.7-macosx10.6.dmg`.
The name of the installer depends on the versions of python and matplotlib it
was built for, and the version of OSX that the matching Python.org installer
was built for.  For example, if the mathing Python.org Python installer was
built for OSX 10.6 or greater, the dmg file will end in ``-macosx10.6.dmg``.
You need to download this disk image file, open the disk image file by double
clicking, and find the new matplotlib disk image icon on your desktop.  Double
click on that icon to show the contents of the image.  Then double-click on
the ``.mpkg`` icon, which will have a name like
:file:`matplotlib-1.4.0-py2.7-macosx10.6.mpkg`, it will run the Installer.app,
prompt you for a password if you need system-wide installation privileges, and
install to a directory like
:file:`/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages`
(exact path depends on your Python version).

Checking your installation
--------------------------

The new version of matplotlib should now be on your Python "path".  Check this
with one of these commands at the Terminal.app command line::

  python2.7 -c 'import matplotlib; print matplotlib.__version__, matplotlib.__file__'

(Python 2.7) or::

  python3.4 -c 'import matplotlib; print(matplotlib.__version__, matplotlib.__file__)'

(Python 3.4).  You should see something like this::

  1.4.0 /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/__init__.pyc

where ``1.4.0`` is the matplotlib version you just installed, and the path
following depends on whether you are using Python.org Python, Homebrew or
Macports.  If you see another version, or you get an error like this::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then check that the Python binary is the one you expected by doing one of
these commands in Terminal.app::

  which python2.7

or::

  which python3.4

If you get the result ``/usr/bin/python2.7``, then you are getting the Python
installed with OSX, which is probably not what you want.  Try closing and
restarting Terminal.app before running the check again. If that doesn't fix
the problem, depending on which Python you wanted to use, consider
reinstalling Python.org Python, or check your homebrew or macports setup.
Remember that the disk image installer only works for Python.org Python, and
will not get picked up by other Pythons.  If all these fail, please let us
know: see :ref:`reporting-problems`.

Windows Notes
=============

We recommend you use one of the excellent python collections which include
Python itself and a wide range of libraries including matplotlib:

- Anaconda_ from `Continuum Analytics`_
- Canopy_ from Enthought_
- `Python (x, y) <https://code.google.com/p/pythonxy>`_

Python (X, Y) is Windows-only, whereas Anaconda and Canopy are cross-platform.

.. _windows-installers:

Standalone binary installers for Windows
----------------------------------------

If you have already installed Python and numpy, you can use one of the
matplotlib binary installers for windows -- you can get these from the
`the PyPI matplotlib page <http://pypi.python.org/pypi/matplotlib>`_
site.  Choose the files with an ``.exe`` extension that match your
version of Python (e.g., ``py2.7`` if you installed Python 2.7).  If
you haven't already installed Python, you can get the official version
from the `Python web site <http://python.org/download/>`_.
