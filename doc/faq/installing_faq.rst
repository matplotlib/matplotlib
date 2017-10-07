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
complexities. Open up a UNIX shell or a DOS command prompt and run, for
example::

   python -c "from pylab import *; plot(); show()" --verbose-helpful

This will give you additional information about which backends matplotlib is
loading, version information, and more. At this point you might want to make
sure you understand matplotlib's :ref:`configuration <sphx_glr_tutorials_introductory_customizing.py>`
process, governed by the :file:`matplotlibrc` configuration file which contains
instructions within and the concept of the matplotlib backend.

If you are still having trouble, see :ref:`reporting-problems`.

.. _clean-install:

How to completely remove Matplotlib
===================================

Occasionally, problems with Matplotlib can be solved with a clean
installation of the package.  In order to fully remove an installed Matplotlib:

1. Delete the caches from your :ref:`Matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Delete any Matplotlib directories or eggs from your :ref:`installation
   directory <locating-matplotlib-install>`.

Linux Notes
===========

To install Matplotlib at the system-level, we recommend that you use your
distribution's package manager.  This will guarantee that Matplotlib's
dependencies will be installed as well.

If, for some reason, you cannot use the package manager, you may use the wheels
available on PyPI::

   python -mpip install matplotlib

or :ref:`build Matplotlib from source <install-from-git>`.

OSX Notes
=========

.. _which-python-for-osx:

Which python for OSX?
---------------------

Apple ships OSX with its own Python, in ``/usr/bin/python``, and its own copy
of Matplotlib. Unfortunately, the way Apple currently installs its own copies
of NumPy, Scipy and Matplotlib means that these packages are difficult to
upgrade (see `system python packages`_).  For that reason we strongly suggest
that you install a fresh version of Python and use that as the basis for
installing libraries such as NumPy and Matplotlib.  One convenient way to
install matplotlib with other useful Python software is to use one of the
excellent Python scientific software collections that are now available:

.. _system python packages:
    https://github.com/MacPython/wiki/wiki/Which-Python#system-python-and-extra-python-packages

- Anaconda_ from `Continuum Analytics`_
- Canopy_ from Enthought_

.. _Canopy: https://www.enthought.com/products/canopy/
.. _Anaconda: https://www.continuum.io/downloads
.. _Enthought: https://www.enthought.com
.. _Continuum Analytics: https://www.continuum.io

These collections include Python itself and a wide range of libraries; if you
need a library that is not available from the collection, you can install it
yourself using standard methods such as *pip*.  Continuum and Enthought offer
their own installation support for these collections; see the Ananconda and
Canopy web pages for more information.

Other options for a fresh Python install are the standard installer from
`python.org <https://www.python.org/downloads/mac-osx/>`_, or installing
Python using a general OSX package management system such as `homebrew
<http://brew.sh>`_ or `macports <https://www.macports.org>`_.  Power users on
OSX will likely want one of homebrew or macports on their system to install
open source software packages, but it is perfectly possible to use these
systems with another source for your Python binary, such as Anaconda, Canopy
or Python.org Python.

.. _install_osx_binaries:

Installing OSX binary wheels
----------------------------

If you are using recent Python from https://www.python.org, Macports or
Homebrew, then you can use the standard pip installer to install Matplotlib
binaries in the form of wheels.

Python.org Python
^^^^^^^^^^^^^^^^^

Install pip following the `standard pip install instructions
<https://pip.readthedocs.io/en/latest/installing/>`_.  For the impatient,
open a new Terminal.app window and::

   curl -O https://bootstrap.pypa.io/get-pip.py

Then (Python 2)::

   python get-pip.py

or (Python 3)::

   python3 get-pip.py

You can now install matplotlib and all its dependencies with ::

   python -mpip install matplotlib

or ::

   python3 -mpip install matplotlib

Macports Python
^^^^^^^^^^^^^^^

For Python 2::

   sudo port install py27-pip
   sudo python2 -mpip install matplotlib

For Python 3::

   sudo port install py36-pip
   sudo python3.6 -mpip install matplotlib

Homebrew Python
^^^^^^^^^^^^^^^

For Python 2::

   python2 -mpip install matplotlib

For Python 3::

   python3 -mpip install matplotlib

You might also want to install IPython or the Jupyter notebook (``pythonX -mpip
install ipython``, ``pythonX -mpip install notebook``, where ``pythonX`` is set
as above).

pip problems
^^^^^^^^^^^^

If you get errors with pip trying to run a compiler like ``gcc`` or ``clang``,
then the first thing to try is to `install xcode
<https://guide.macports.org/chunked/installing.html#installing.xcode>`_ and
retry the install.  If that does not work, then check
:ref:`reporting-problems`.

Checking your installation
--------------------------

The new version of Matplotlib should now be on your Python "path".  Check this
with one of these commands at the Terminal.app command line::

  python2 -c 'import matplotlib; print matplotlib.__version__, matplotlib.__file__'

(Python 2) or::

  python3 -c 'import matplotlib; print(matplotlib.__version__, matplotlib.__file__)'

(Python 3).  You should see something like this::

  2.1.0 /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/__init__.pyc

where ``2.1.0`` is the Matplotlib version you just installed, and the path
following depends on whether you are using Python.org Python, Homebrew or
Macports.  If you see another version, or you get an error like this::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then check that the Python binary is the one you expected by doing one of
these commands in Terminal.app::

  which python2

or::

  which python3

If you get the result ``/usr/bin/python2.7``, then you are getting the Python
installed with OSX, which is probably not what you want.  Try closing and
restarting Terminal.app before running the check again. If that doesn't fix the
problem, depending on which Python you wanted to use, consider reinstalling
Python.org Python, or check your homebrew or macports setup.  Remember that
the disk image installer only works for Python.org Python, and will not get
picked up by other Pythons.  If all these fail, please :ref:`let us know
<reporting-problems>`.

Windows Notes
=============

See :ref:`installing_windows`.

.. _install-from-git:

Install from source
===================

Clone the main source using one of::

   git clone git@github.com:matplotlib/matplotlib.git

or::

   git clone git://github.com/matplotlib/matplotlib.git

and build and install as usual with::

   cd matplotlib
   python -mpip install .

.. note::

   If you are on Debian/Ubuntu, you can get all the dependencies required to
   build Matplotlib with::

      sudo apt-get build-dep python-matplotlib

   If you are on Fedora/RedHat, you can get all the dependencies required to
   build matplotlib by first installing ``yum-builddep`` and then running::

      su -c 'yum-builddep python-matplotlib'

   This does not build Matplotlib, but it does get all of the build
   dependencies, which will make building from source easier.

If you want to be able to follow the development branch as it changes
just replace the last step with::

   python -mpip install -e .

This creates links and installs the command line script in the appropriate
places.

.. note::
   OSX users please see the :ref:`build_osx` guide.

   Windows users please see the :ref:`build_windows` guide.

Then, if you want to update your matplotlib at any time, just do::

   git pull

When you run ``git pull``, if the output shows that only Python files have
been updated, you are all set. If C files have changed, you need to run ``pip
install -e .`` again to compile them.

There is more information on :ref:`using git <using-git>` in the developer
docs.
