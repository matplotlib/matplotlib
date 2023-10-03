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
