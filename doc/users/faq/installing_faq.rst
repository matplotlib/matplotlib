.. _installing-faq:

.. redirect-from:: /faq/installing_faq

**************
Installing FAQ
**************

.. contents::
   :backlinks: none

Report a compilation problem
============================

See :ref:`reporting-problems`.

Matplotlib compiled fine, but nothing shows up when I use it
============================================================

The first thing to try is a :ref:`clean install <clean-install>` and see if
that helps.  If not, the best way to test your install is by running a script,
rather than working interactively from a python shell or an integrated
development environment such as :program:`IDLE` which add additional
complexities. Open up a UNIX shell or a DOS command prompt and run, for
example::

   python -c "from pylab import *; set_loglevel('debug'); plot(); show()"

This will give you additional information about which backends Matplotlib is
loading, version information, and more. At this point you might want to make
sure you understand Matplotlib's :doc:`configuration </tutorials/introductory/customizing>`
process, governed by the :file:`matplotlibrc` configuration file which contains
instructions within and the concept of the Matplotlib backend.

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
install Matplotlib with other useful Python software is to use the Anaconda_
Python scientific software collection, which includes Python itself and a
wide range of libraries; if you need a library that is not available from the
collection, you can install it yourself using standard methods such as *pip*.
See the Ananconda web page for installation support.

.. _system python packages:
    https://github.com/MacPython/wiki/wiki/Which-Python#system-python-and-extra-python-packages
.. _Anaconda: https://www.anaconda.com/

Other options for a fresh Python install are the standard installer from
`python.org <https://www.python.org/downloads/mac-osx/>`_, or installing
Python using a general OSX package management system such as `homebrew
<https://brew.sh/>`_ or `macports <https://www.macports.org>`_.  Power users on
OSX will likely want one of homebrew or macports on their system to install
open source software packages, but it is perfectly possible to use these
systems with another source for your Python binary, such as Anaconda
or Python.org Python.

.. _install_osx_binaries:

Installing OSX binary wheels
----------------------------

If you are using Python from https://www.python.org, Homebrew, or Macports,
then you can use the standard pip installer to install Matplotlib binaries in
the form of wheels.

pip is installed by default with python.org and Homebrew Python, but needs to
be manually installed on Macports with ::

   sudo port install py38-pip

Once pip is installed, you can install Matplotlib and all its dependencies with
from the Terminal.app command line::

   python3 -mpip install matplotlib

(``sudo python3.6 ...`` on Macports).

You might also want to install IPython or the Jupyter notebook (``python3 -mpip
install ipython notebook``).

Checking your installation
--------------------------

The new version of Matplotlib should now be on your Python "path".  Check this
at the Terminal.app command line::

  python3 -c 'import matplotlib; print(matplotlib.__version__, matplotlib.__file__)'

You should see something like ::

  3.0.0 /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/__init__.py

where ``3.0.0`` is the Matplotlib version you just installed, and the path
following depends on whether you are using Python.org Python, Homebrew or
Macports.  If you see another version, or you get an error like ::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then check that the Python binary is the one you expected by running ::

  which python3

If you get a result like ``/usr/bin/python...``, then you are getting the
Python installed with OSX, which is probably not what you want.  Try closing
and restarting Terminal.app before running the check again. If that doesn't fix
the problem, depending on which Python you wanted to use, consider reinstalling
Python.org Python, or check your homebrew or macports setup.  Remember that
the disk image installer only works for Python.org Python, and will not get
picked up by other Pythons.  If all these fail, please :ref:`let us know
<reporting-problems>`.
