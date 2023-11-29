.. _troubleshooting-faq:

.. redirect-from:: /faq/troubleshooting_faq
.. redirect-from:: /users/faq/troubleshooting_faq

===============
Troubleshooting
===============

For guidance on debugging an installation, see :ref:`installing-faq`.


.. _git-trouble:

Problems with git
=================

First, make sure you have a clean build and install (see :ref:`clean-install`),
get the latest git update, install it and run a simple test script in debug
mode::

    rm -rf /path/to/site-packages/matplotlib*
    git clean -xfd
    git pull
    python -m pip install -v . > build.out
    python -c "from pylab import *; set_loglevel('debug'); plot(); show()" > run.out

and post :file:`build.out` and :file:`run.out` to the `matplotlib-devel
<https://mail.python.org/mailman/listinfo/matplotlib-devel>`_
mailing list (please do not post git problems to the `users list
<https://mail.python.org/mailman/listinfo/matplotlib-users>`_).

Of course, you will want to clearly describe your problem, what you
are expecting and what you are getting, but often a clean build and
install will help.  See also :ref:`reporting-problems`.

Unlink of file ``*/_c_internal_utils.cp311-win_amd64.pyd`` failed
============================================================================

The DLL files may be loaded by multiple running instances of Matplotlib; therefore
check that Matplotlib is not running in any other application before trying to
unlink this file. Multiple versions of Matplotlib can be linked to the same DLL,
for example a development version installed in a development conda environment
and a stable version running in a Jupyter notebook. To resolve this error, fully
close all running instances of Matplotlib.

Windows compilation errors
==========================
If the compiled extensions are not building on Windows due to errors in linking to
Windows' header files, for example ``../../src/_tkagg.cpp:133:10: error: 'WM_DPICHANGED' was not declared in this scope``,
you should check which compiler Meson is using:

.. code-block:: bat

    Build type: native build
    Project name: matplotlib
    Project version: 3.9.0.dev0
    C compiler for the host machine: cc (gcc 7.2.0 "cc (Rev1, Built by MSYS2 project) 7.2.0")
    C linker for the host machine: cc ld.bfd 2.29.1
    C++ compiler for the host machine: c++ (gcc 7.2.0 "c++ (Rev1, Built by MSYS2 project) 7.2.0")
    C++ linker for the host machine: c++ ld.bfd 2.29.1

Our :ref:`dependencies <dependencies>` documentation lists the minimum header
version if you intended to use ``MSYS2``. If you intended to use ``MSVC`` then
you may need to force Meson to :external+meson-python:ref:`use MSVC <vsenv-example>`.
