.. redirect-from:: /users/installing

############
Installation
############

==============================
Installing an official release
==============================

Matplotlib releases are available as wheel packages for macOS, Windows and
Linux on `PyPI <https://pypi.org/project/matplotlib/>`_. Install it using
``pip``:

.. code-block:: sh

  python -m pip install -U pip
  python -m pip install -U matplotlib

If this command results in Matplotlib being compiled from source and
there's trouble with the compilation, you can add ``--prefer-binary`` to
select the newest version of Matplotlib for which there is a
precompiled wheel for your OS and Python.

.. note::

   The following backends work out of the box: Agg, ps, pdf, svg

   Python is typically shipped with tk bindings which are used by
   TkAgg.

   For support of other GUI frameworks, LaTeX rendering, saving
   animations and a larger selection of file formats, you can
   install :ref:`optional_dependencies`.

=========================
Third-party distributions
=========================

Various third-parties provide Matplotlib for their environments.

Conda packages
==============
Matplotlib is available both via the *anaconda main channel*

.. code-block:: sh

   conda install matplotlib

as well as via the *conda-forge community channel*

.. code-block:: sh

   conda install -c conda-forge matplotlib

Python distributions
====================

Matplotlib is part of major Python distributions:

- `Anaconda <https://www.anaconda.com/>`_

- `ActiveState ActivePython
  <https://www.activestate.com/products/python/downloads/>`_

- `WinPython <https://winpython.github.io/>`_

Linux package manager
=====================

If you are using the Python version that comes with your Linux distribution,
you can install Matplotlib via your package manager, e.g.:

* Debian / Ubuntu: ``sudo apt-get install python3-matplotlib``
* Fedora: ``sudo dnf install python3-matplotlib``
* Red Hat: ``sudo yum install python3-matplotlib``
* Arch: ``sudo pacman -S python-matplotlib``

.. redirect-from:: /users/installing/installing_source

.. _install_from_source:

==========================
Installing a nightly build
==========================

Matplotlib makes nightly development build wheels available on the
`scipy-wheels-nightly Anaconda Cloud organization
<https://anaconda.org/scipy-wheels-nightly>`_.
These wheels can be installed with ``pip`` by specifying scipy-wheels-nightly
as the package index to query:

.. code-block:: sh

  python -m pip install \
    --upgrade \
    --pre \
    --index-url https://pypi.anaconda.org/scipy-wheels-nightly/simple \
    --extra-index-url https://pypi.org/simple \
    matplotlib

======================
Installing from source
======================

If you are interested in contributing to Matplotlib development,
running the latest source code, or just like to build everything
yourself, it is not difficult to build Matplotlib from source.

First you need to install the :ref:`dependencies`.

A C compiler is required.  Typically, on Linux, you will need ``gcc``, which
should be installed using your distribution's package manager; on macOS, you
will need xcode_; on Windows, you will need `Visual Studio`_ 2015 or later.

For those using Visual Studio, make sure "Desktop development with C++" is
selected, and that the latest MSVC, "C++ CMake tools for Windows," and a
Windows SDK compatible with your version of Windows are selected and installed.
They should be selected by default under the "Optional" subheading, but are
required to build matplotlib from source.

.. _xcode: https://guide.macports.org/chunked/installing.html#installing.xcode

.. _Visual Studio: https://visualstudio.microsoft.com/downloads/

The easiest way to get the latest development version to start contributing
is to go to the git `repository <https://github.com/matplotlib/matplotlib>`_
and run::

  git clone https://github.com/matplotlib/matplotlib.git

or::

  git clone git@github.com:matplotlib/matplotlib.git

If you're developing, it's better to do it in editable mode. The reason why
is that pytest's test discovery only works for Matplotlib
if installation is done this way. Also, editable mode allows your code changes
to be instantly propagated to your library code without reinstalling (though
you will have to restart your python process / kernel)::

  cd matplotlib
  python -m pip install -e .

If you're not developing, it can be installed from the source directory with
a simple (just replace the last step)::

  python -m pip install .

To run the tests you will need to install some additional dependencies::

  python -m pip install -r requirements/dev/dev-requirements.txt

Then, if you want to update your Matplotlib at any time, just do::

  git pull

When you run ``git pull``, if the output shows that only Python files have
been updated, you are all set. If C files have changed, you need to run ``pip
install -e .`` again to compile them.

There is more information on :ref:`using git <using-git>` in the developer
docs.

.. warning::

  The following instructions in this section are for very custom
  installations of Matplotlib. Proceed with caution because these instructions
  may result in your build producing unexpected behavior and/or causing
  local testing to fail.

If you would like to build from a tarball, grab the latest *tar.gz* release
file from `the PyPI files page <https://pypi.org/project/matplotlib/>`_.

We provide a `mplsetup.cfg`_ file which you can use to customize the build
process. For example, which default backend to use, whether some of the
optional libraries that Matplotlib ships with are installed, and so on.  This
file will be particularly useful to those packaging Matplotlib.

.. _mplsetup.cfg: https://raw.githubusercontent.com/matplotlib/matplotlib/main/mplsetup.cfg.template

If you are building your own Matplotlib wheels (or sdists) on Windows, note
that any DLLs that you copy into the source tree will be packaged too.

==========================
Installing for development
==========================
See :ref:`installing_for_devs`.

.. redirect-from:: /faq/installing_faq
.. redirect-from:: /users/faq/installing_faq

.. _installing-faq:

==========================
Frequently asked questions
==========================

.. contents::
   :backlinks: none
   :local:

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
See the Anaconda web page for installation support.

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

   python3 -m pip install matplotlib

You might also want to install IPython or the Jupyter notebook (``python3 -m pip
install ipython notebook``).

Checking your installation
--------------------------

The new version of Matplotlib should now be on your Python "path".  Check this
at the Terminal.app command line::

  python3 -c 'import matplotlib; print(matplotlib.__version__, matplotlib.__file__)'

You should see something like ::

  3.6.0 /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/matplotlib/__init__.py

where ``3.6.0`` is the Matplotlib version you just installed, and the path
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
