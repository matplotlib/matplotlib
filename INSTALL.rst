==================
Installation Guide
==================

.. note::

    If you wish to contribute to the project, it's recommended you
    :ref:`install the latest development version<install_from_source>`.

.. contents::

Installing an official release
==============================

Matplotlib and its dependencies are available as wheel packages for macOS,
Windows and Linux distributions::

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

Although not required, we suggest also installing ``IPython`` for
interactive use.  To easily install a complete Scientific Python
stack, see :ref:`install_scipy_dists` below.

Third-party distributions of Matplotlib
=======================================

.. _install_scipy_dists:

Scientific Python Distributions
-------------------------------

`Anaconda <https://www.anaconda.com/>`_ and `ActiveState
<https://www.activestate.com/activepython/downloads>`_ are excellent
choices that "just work" out of the box for Windows, macOS and common
Linux platforms. `WinPython <https://winpython.github.io/>`_ is an
option for Windows users.  All of these distributions include
Matplotlib and *lots* of other useful (data) science tools.

Linux: using your package manager
---------------------------------

If you are on Linux, you might prefer to use your package manager.  Matplotlib
is packaged for almost every major Linux distribution.

* Debian / Ubuntu: ``sudo apt-get install python3-matplotlib``
* Fedora: ``sudo dnf install python3-matplotlib``
* Red Hat: ``sudo yum install python3-matplotlib``
* Arch: ``sudo pacman -S python-matplotlib``

.. _install_from_source:

Installing from source
======================

If you are interested in contributing to Matplotlib development,
running the latest source code, or just like to build everything
yourself, it is not difficult to build Matplotlib from source.

First you need to install the :ref:`dependencies`.

A C compiler is required.  Typically, on Linux, you will need ``gcc``, which
should be installed using your distribution's package manager; on macOS, you
will need xcode_; on Windows, you will need Visual Studio 2015 or later.

.. _xcode: https://guide.macports.org/chunked/installing.html#installing.xcode

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

We provide a `setup.cfg`_ file which you can use to customize the build
process. For example, which default backend to use, whether some of the
optional libraries that Matplotlib ships with are installed, and so on.  This
file will be particularly useful to those packaging Matplotlib.

.. _setup.cfg: https://raw.githubusercontent.com/matplotlib/matplotlib/master/setup.cfg.template

Building on Windows
-------------------

Compiling Matplotlib (or any other extension module, for that matter) requires
Visual Studio 2015 or later.

If you are building your own Matplotlib wheels (or sdists), note that any DLLs
that you copy into the source tree will be packaged too.
