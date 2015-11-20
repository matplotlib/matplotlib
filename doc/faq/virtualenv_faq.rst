.. _virtualenv-faq:

***********************************************
Working with Matplotlib in Virtual environments
***********************************************

.. contents::
   :backlinks: none


.. _introduction:

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
* The upcoming `WX Phoenix <http://wiki.wxpython.org/ProjectPhoenix>`_ toolkit
  is ``pip`` installable.

Other frameworks are harder to install into a virtual environment. There are at
least two possible ways to get access to these in a virtual environment.

One often suggested solution is to use the ``--system-site-packages`` option
to virtualenv when creating an environment. This adds all system wide packages
to the virtual environment. However, this breaks the isolation between the
virtual environment and the system install. Among other issues it results in
hard to debug problems with system packages shadowing the environment packages.
If you use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/>`_
this can be toggled with the ``toggleglobalsitepackages`` command.

Alternatively, you can manually symlink the GUI frameworks into the environment.
I.e. to use PyQt5, you should symlink ``PyQt5`` and ``sip`` from your system
site packages directory into the environment taking care that the environment
and the systemwide install use the same python version.

OSX
===

On OSX, two different types of Python Builds exist: a regular build and a
framework build. In order to interact correctly with OSX through some
GUI frameworks you need a framework build of Python.
At the time of writing the ``macosx``, ``WX`` and ``WXAgg`` backends require a
framework build to function correctly. Unfortunately virtualenv creates a non
framework build even if created from a framework build of Python. Conda
environments are framework builds. From
Matplotlib 1.5 onwards the ``macosx`` backend checks that a framework build is
available and fails if a non framework build is found.
WX has a similar check build in.

The issue has been reported on the virtualenv bug tracker `here
<https://github.com/pypa/virtualenv/issues/54>`__ and `here
<https://github.com/pypa/virtualenv/issues/609>`__

Until this is fixed, one of the following workarounds must be used:

``PYTHONHOME`` Script
---------------------

The best known workaround,
borrowed  from the `WX wiki
<http://wiki.wxpython.org/wxPythonVirtualenvOnMac>`_, is to  use the non
virtualenv python along with the PYTHONHOME environment variable.  This can be
implemented in a script as below. To use this modify ``PYVER`` and
``PATHTOPYTHON`` and put the script in the virtualenv bin directory i.e.
``PATHTOVENV/bin/frameworkpython``

.. code:: bash

  #!/bin/bash

  # what real Python executable to use
  PYVER=2.7
  PATHTOPYTHON=/usr/local/bin/
  PYTHON=${PATHTOPYTHON}python${PYVER}

  # find the root of the virtualenv, it should be the parent of the dir this script is in
  ENV=`$PYTHON -c "import os; print os.path.abspath(os.path.join(os.path.dirname(\"$0\"), '..'))"`

  # now run Python with the virtualenv set as Python's HOME
  export PYTHONHOME=$ENV
  exec $PYTHON "$@"


With this in place you can run ``frameworkpython`` to get an interactive
framework build within the virtualenv. To run a script you can do
``frameworkpython test.py`` where ``test.py`` is a script that requires a
framework build. To run an interactive ``IPython`` session with the framework
build within the virtual environment you can do ``frameworkpython -m IPython``

``PYTHONHOME`` Function
-----------------------

Alternatively you can define a function in your ``.bashrc`` using

.. code:: bash

  function frameworkpython {
      if [[ ! -z "$VIRTUAL_ENV" ]]; then
          PYTHONHOME=$VIRTUAL_ENV /usr/local/bin/python "$@"
      else
          /usr/local/bin/python "$@"
      fi
  }

This function can then be used in all of your virtualenvs without having to
fix every single one of them.

PythonW Compiler
----------------

In addition
`virtualenv-pythonw-osx <https://github.com/gldnspud/virtualenv-pythonw-osx>`_
provides an alternative workaround which may be used to solve the issue.
