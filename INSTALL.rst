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

======================
Installing from source
======================
See :ref:`install_from_source`.

==========================
Installing for development
==========================
See :ref:`installing_for_devs`.

==============
Installing FAQ
==============

See :ref:`installing-faq`.
