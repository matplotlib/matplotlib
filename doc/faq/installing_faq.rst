.. _installing-faq:

*****************
 Installation FAQ
*****************


.. contents::
   :backlinks: none


Report a compilation problem
======================================

See :ref:`reporting-problems`.

matplotlib compiled fine, but nothing shows up with plot
==========================================================

The first thing to try is a :ref:`clean install <clean-install>` and see if
that helps.  If not, the best way to test your install is by running a script,
rather than working interactively from a python shell or an integrated
development environment such as :program:`IDLE` which add additional
complexities. Open up a UNIX shell or a DOS command prompt and cd into a
directory containing a minimal example in a file. Something like
:file:`simple_plot.py`, or for example::

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

Cleanly rebuild and reinstall everything
========================================

The steps depend on your platform and installation method.

Easy Install
------------

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Run::

     easy_install -m PackageName


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

2. Delete the ``build`` directory in the source tree

3. Delete any matplotlib directories or eggs from your `installation directory
   <locating-matplotlib-install>`


.. _install-from-git:

Install from git
================

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


If you want to be able to follow the development branch as it changes just replace
the last step with (make sure you have **setuptools** installed)::

  > python setupegg.py develop

This creates links in the right places and installs the command line script to the appropriate places.
Then, if you want to update your **matplotlib** at any time, just do::

  > git pull

When you run `git pull`, if the output shows that only Python files have been
updated, you are all set. If C files have changed, you need to run the `python
setupegg.py develop` command again to compile them.

There is more information on :ref:`using git <using-git>` in
the developer docs.


OS-X questions
==============

.. _which-python-for-osx:

Which python for OS X?
----------------------

Apple ships with its own python, many users have had trouble
with it so there are alternatives.  If it is feasible for you, we
recommend the enthought python distribution `EPD
<http://www.enthought.com/products/epd.php>`_ for OS X (which comes
with matplotlib and much more) or the
`MacPython <http://wiki.python.org/moin/MacPython/Leopard>`_ or the
official OS X version from `python.org
<http://www.python.org/download/>`_.


.. _install_osx_binaries:

Installing OSX binaries
-----------------------

If you want to install matplotlib from one of the binary installers we
build, you have two choices: a mpkg installer, which is a typical
Installer.app, or an binary OSX egg, which you can install via
setuptools easy_install.

The mkpg installer will have a "zip" extension, and will have a name
like file:`matplotlib-0.99.0.rc1-py2.5-macosx10.5_mpkg.zip` depending on
the python, matplotlib, and OSX versions.  You need to unzip this file
using either the "unzip" command on OSX, or simply double clicking on
it to run StuffIt Expander.  When you double click on the resultant
mpkd directory, which will have a name like
file:`matplotlib-0.99.0.rc1-py2.5-macosx10.5.mpkg`, it will run the
Installer.app, prompt you for a password if you need system wide
installation privileges, and install to a directory like
file:`/Library/Python/2.5/site-packages/`, again depending on your
python version.  This directory may not be in your python path, so you
should test your installation with::

  > python -c 'import matplotlib; print matplotlib.__version__, matplotlib.__file__'

If you get an error like::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then you will need to set your PYTHONPATH, eg::

    export PYTHONPATH=/Library/Python/2.5/site-packages:$PYTHONPATH

See also ref:`environment-variables`.

.. _easy-install-osx-egg:

easy_install from egg
------------------------------

You can also use the eggs we build for OSX (see the `installation
instructions
<http://pypi.python.org/pypi/setuptools#cygwin-mac-os-x-linux-other>`_
for easy_install if you do not have it on your system already).  You
can try::

    > easy_install matplotlib

which should grab the latest egg from the sourceforge site, but the
naming conventions for OSX eggs appear to be broken (see below) so
there is no guarantee the right egg will be found.  We recommend you
download the latest egg from our `download site
<http://sourceforge.net/projects/matplotlib/files/>`_ directly to your
harddrive, and manually install it with

    > easy_install --install-dir=~/dev/lib/python2.5/site-packages/  matplotlib-0.99.0.rc1-py2.5-macosx-10.5-i386.egg


Some users have reported problems with the egg for 0.98 from the
matplotlib download site, with ``easy_install``, getting an error::

    > easy_install ./matplotlib-0.98.0-py2.5-macosx-10.3-fat.egg
    Processing matplotlib-0.98.0-py2.5-macosx-10.3-fat.egg
    removing '/Library/Python/2.5/site-packages/matplotlib-0.98.0-py2.5-
    ...snip...
    Reading http://matplotlib.sourceforge.net
    Reading http://cheeseshop.python.org/pypi/matplotlib/0.91.3
    No local packages or download links found for matplotlib==0.98.0
    error: Could not find suitable distribution for
    Requirement.parse('matplotlib==0.98.0')

If you rename ``matplotlib-0.98.0-py2.5-macosx-10.3-fat.egg`` to
``matplotlib-0.98.0-py2.5.egg``, ``easy_install`` will install it from
the disk.  Many Mac OS X eggs have cruft at the end of the filename,
which prevents their installation through easy_install.  Renaming is
all it takes to install them; still, it's annoying.


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

3. Save the following as a shell script , for example
``./install-matplotlib-epd-osx.sh``::

   NAME=matplotlib
   VERSION=v1.0.x
   PREFIX=$HOME
   #branch="release"
   branch="trunk"
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
checks out the source from github, builds and installs it. The backend seems
to be set to MacOSX.


Windows questions
=================

.. _windows-installers:

Binary installers for windows
----------------------------------------------

If you have already installed python, you can use one of the
matplotlib binary installers for windows -- you can get these from the
`sourceforge download
<http://sourceforge.net/project/platformdownload.php?group_id=80706>`_
site.  Choose the files that match your version of python (eg
``py2.5`` if you installed Python 2.5) which have the ``exe``
extension.  If you haven't already installed python, you can get the
official version from the `python web site
<http://python.org/download/>`_.  There are also two packaged
distributions of python that come preloaded with matplotlib and many
other tools like ipython, numpy, scipy, vtk and user interface
toolkits.  These packages are quite large because they come with so
much, but you get everything with a single click installer.

* the enthought python distribution `EPD
  <http://www.enthought.com/products/epd.php>`_

* `python (x, y) <http://www.pythonxy.com/foreword.php>`_
