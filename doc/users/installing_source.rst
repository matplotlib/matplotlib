.. _install_from_source:

======================
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

We provide a `mplsetup.cfg`_ file which you can use to customize the build
process. For example, which default backend to use, whether some of the
optional libraries that Matplotlib ships with are installed, and so on.  This
file will be particularly useful to those packaging Matplotlib.

.. _mplsetup.cfg: https://raw.githubusercontent.com/matplotlib/matplotlib/master/mplsetup.cfg.template

If you are building your own Matplotlib wheels (or sdists) on Windows, note
that any DLLs that you copy into the source tree will be packaged too.
