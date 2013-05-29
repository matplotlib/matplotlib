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
setupegg.py develop` command again to compile them.

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

Apple ships with its own python, and many users have had trouble
with it. There are several alternative versions of python that
can be used. If it is feasible, we recommend that you use the enthought
python distribution `EPD <http://www.enthought.com/products/epd.php>`_
for OS X (which comes with matplotlib and much more). Also available is
`MacPython <http://wiki.python.org/moin/MacPython/Leopard>`_ or the
official OS X version from `python.org <http://www.python.org/download/>`_.

.. note::
   Before installing any of the binary packages, be sure that all of the
   packages were compiled for the same version of python.
   Often, the download site for NumPy and matplotlib will display a
   supposed 'current' version of the package, but you may need to choose
   a different package from the full list that was built for your
   combination of python and OSX.


.. _install_osx_binaries:

Installing OSX binaries
-----------------------

If you want to install matplotlib from one of the binary installers we
build, you have two choices: a mpkg installer, which is a typical
Installer.app, or a binary OSX egg, which you can install via
setuptools' easy_install.

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

.. _install_from_source_on_osx_epd:

Building and installing from source on OSX with EPD
---------------------------------------------------

If you have the EPD installed (:ref:`which-python-for-osx`), it might turn out
to be rather tricky to install a new version of matplotlib from source on the
Mac OS 10.5 . Here's a procedure that seems to work, at least sometimes:

0. Remove the ~/.matplotlib folder ("rm -rf ~/.matplotlib").

1. Edit the file (make a backup before you start, just in case):
``/Library/Frameworks/Python.framework/Versions/Current/lib/python2.5/config/Makefile``,
removing all occurrences of the string ``-arch ppc``, changing the line
``MACOSX_DEPLOYMENT_TARGET=10.3`` to ``MACOSX_DEPLOYMENT_TARGET=10.5`` and
changing the occurrences of ``MacOSX10.4u.sdk`` into ``MacOSX10.5.sdk``

2.  In
``/Library/Frameworks/Python.framework/Versions/Current/lib/pythonX.Y/site-packages/easy-install.pth``,
(where X.Y is the version of Python you are building against)
Comment out the line containing the name of the directory in which the
previous version of MPL was installed (Looks something like
``./matplotlib-0.98.5.2n2-py2.5-macosx-10.3-fat.egg``).

3. Save the following as a shell script, for example
``./install-matplotlib-epd-osx.sh``::

   NAME=matplotlib
   VERSION=v1.1.x
   PREFIX=$HOME
   #branch="release"
   branch="master"
   git clone git://github.com/matplotlib/matplotlib.git
   cd matplotlib
   if [ $branch = "release" ]
       then
       echo getting the maintenance branch
       git checkout -b $VERSION origin/$VERSION
   fi
   export CFLAGS="-Os -arch i386"
   export LDFLAGS="-Os -arch i386"
   export PKG_CONFIG_PATH="/usr/x11/lib/pkgconfig"
   export ARCHFLAGS="-arch i386"
   python setup.py build
   # use --prefix if you don't want it installed in the default location:
   python setup.py install #--prefix=$PREFIX
   cd ..

Run this script (for example ``sh ./install-matplotlib-epd-osx.sh``) in the
directory in which you want the source code to be placed, or simply type the
commands in the terminal command line. This script sets some local variable
(CFLAGS, LDFLAGS, PKG_CONFIG_PATH, ARCHFLAGS), removes previous installations,
checks out the source from github, builds and installs it. The backend should
to be set to MacOSX.


Windows Notes
=============

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

There are also two packaged distributions of python that come
preloaded with matplotlib and many other tools like ipython, numpy,
scipy, vtk and user interface toolkits.  These packages are quite
large because they come with so much, but you get everything with
a single click installer.

* The Enthought Python Distribution `EPD
  <http://www.enthought.com/products/epd.php>`_

* `python (x, y) <http://www.pythonxy.com>`_
