############
Installation
############

.. toctree::
   :hidden:

   installing_source.rst


==============================
Installing an official release
==============================

Matplotlib releases are available as wheel packages for macOS, Windows and
Linux on `PyPI <https://pypi.org/project/matplotlib/>`_. Install it using
``pip``:

.. code-block:: sh

  python -m pip install -U pip
  python -m pip install -U matplotlib

While we make a best effort to publish wheels concurrently with the source,
there can be a lag in availability or coverage for you platform, OS, or Python
version.  If ``pip`` can not find a wheel suitable for your system it will by
default try to :ref:`compile Matplotlib from source <install_from_source>`
which requires a compiler and the Python headers which are not available on all
systems.  To be sure that you will always get a pre-compiled version of
Matplotlib use the ``--only-binary``
`option <https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-only-binary>`__ to
``pip``

.. code-block:: sh

  python -m pip install -U matplotlib --only-binary matplotlib

or specify Matplotlib in your `requirements file
<https://pip.pypa.io/en/stable/user_guide/#requirements-files>`_ as

.. code-block:: sh

   matplotlib --only-binary matplotlib

You can also use the ``--prefer-binary``
`option <https://pip.pypa.io/en/stable/cli/pip_install/#install-prefer-binary>`__ which
will cause ``pip`` to pick older wheels over newer sdists for all packages.

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

======================
Installing from source
======================
See :ref:`install_from_source`.

==========================
Installing for development
==========================
See :ref:`installing_for_devs`.
