.. _virtualenv-faq:

***********************************************
Working with Matplotlib in Virtual environments
***********************************************

.. contents::
   :backlinks: none


.. _virtualenv_introduction:

Introduction
============

When running :mod:`matplotlib` in a
`virtual environment <https://virtualenv.pypa.io/en/latest/>`_ you may discover
a few issues. :mod:`matplotlib` itself has no issue with virtual environments.
However, the GUI frameworks that :mod:`matplotlib` uses for interactive
figures have some issues with virtual environments. Everything below assumes
some familiarity with the Matplotlib backends as found in :ref:`What is a
backend? <what-is-a-backend>`.

If you only use the ``IPython/Jupyter Notebook``'s ``inline`` and ``notebook``
backends and non interactive backends you should not have any issues and can
ignore everything below.

If you are using Matplotlib on OSX you may also want to consider the
:ref:`OSX framework FAQ <osxframework-faq>`.

GUI Frameworks
==============

Interactive Matplotlib relies heavily on the interaction with external GUI
frameworks.

Most GUI frameworks are not pip installable. This makes it tricky to install
them within a virtual environment. This problem does not exist if you use Conda
environments where you can install all Conda supported GUI frameworks directly
into the environment. In regular virtualenv environment various workarounds
exist. Some of these are given here:

* The ``TKAgg`` backend doesn't require any external dependencies and is
  normally always available.
* The ``QT4`` framework ``PySide`` is pip installable.
* ``PYQT5`` is pip installable on Python 3.5.

Other frameworks are harder to install into a virtual environment. There are at
least two possible ways to get access to these in a virtual environment.

One often suggested solution is to use the ``--system-site-packages`` option
to virtualenv when creating an environment. This adds all system wide packages
to the virtual environment. However, this breaks the isolation between the
virtual environment and the system install. Among other issues it results in
hard to debug problems with system packages shadowing the environment packages.
If you use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/>`_
this can be toggled with the ``toggleglobalsitepackages`` command.

Alternatively, you can manually symlink the GUI frameworks into the environment.
I.e. to use PyQt5, you should symlink ``PyQt5`` and ``sip`` from your system
site packages directory into the environment taking care that the environment
and the systemwide install use the same python version.
