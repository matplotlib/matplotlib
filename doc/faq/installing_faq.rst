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

Apple ships OS X with its own python, but it is generally recommended
that users install an independent python system and use that
as the basis for installing libraries such as numpy and
matplotlib.  By far the easiest method is to use one
of the excellent python software collections that are now
available:

- Anaconda_ from `Continuum Analytics`_
- Canopy_ from Enthought_

.. _Canopy: https://enthought.com/products/canopy/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Enthought: http://enthought.com
.. _Continuum Analytics: http://continuum.io

These collections include python itself and a wide range of
libraries; if you need a library that is not available from
the collection, you can install it yourself using standard
methods such as *pip*.

If you choose not to use a collection, then you may use an
installer from `python.org
<https://www.python.org/downloads/mac-osx/>`_, or use a
general package management system such as `homebrew
<http://brew.sh>`_ or `macports <http://www.macports.org>`_.
Whatever you choose, choose one and stick with it--don't try
to mix homebrew and macports, for example.  You may,
however, use homebrew or macports for all your non-python
software, and still use Anaconda_ or Canopy_ for your python
system.



.. _install_osx_binaries:

Installing OSX binaries
-----------------------

If you want to install matplotlib from one of the binary installers we
build, you have two choices: a mpkg installer, which is a typical
Installer.app, or a binary OSX egg, which you can install via
setuptools' easy_install.

.. note::
   Before installing any of the binary packages, be sure that all of the
   packages were compiled for the same version of python.
   Often, the download site for NumPy and matplotlib will display a
   supposed 'current' version of the package, but you may need to choose
   a different package from the full list that was built for your
   combination of python and OSX.

The mkpg installer will have a "zip" extension, and will have a name
like :file:`matplotlib-1.2.0-py2.7-macosx10.5_mpkg.zip`.
The name of the installer depends on which versions of python, matplotlib,
and OSX it was built for.  You need to unzip this file using either the
"unzip" command, or simply double clicking on the it. Then when you
double-click on the resulting mpkd, which will have a name like
:file:`matplotlib-1.2.0-py2.7-macosx10.5.mpkg`, it will run the
Installer.app, prompt you for a password if you need system-wide
installation privileges, and install to a directory like
:file:`/Library/Python/2.7/site-packages/` (exact path depends on your
python version).  This directory may not be in your python 'path' variable,
so you should test your installation with::

  > python -c 'import matplotlib; print matplotlib.__version__, matplotlib.__file__'

If you get an error like::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then you will need to set your PYTHONPATH, eg::

    export PYTHONPATH=/Library/Python/2.7/site-packages:$PYTHONPATH

See also ref:`environment-variables`.


Windows Notes
=============

Recommendation: use one of the excellent multi-platform
python collections which include python itself and a wide
range of libraries including matplotlib:

- Anaconda_ from `Continuum Analytics`_
- Canopy_ from Enthought_

A Windows-only alternative is:

- `python (x, y) <http://www.pythonxy.com>`_

.. _windows-installers:

Binary installers for Windows
-----------------------------

If you have already installed python, you can use one of the
matplotlib binary installers for windows -- you can get these from the
`download <http://matplotlib.org/downloads.html>`_ site.
Choose the files that match your version of python (eg ``py2.7`` if
you installed Python 2.7) which have the ``exe`` extension.  If you
haven't already installed python, you can get the official version
from the `python web site <http://python.org/download/>`_.

