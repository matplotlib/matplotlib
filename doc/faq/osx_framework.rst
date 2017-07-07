.. _osxframework-faq:

******************************
Working with Matplotlib on OSX
******************************

.. contents::
   :backlinks: none


.. _osxframework_introduction:

Introduction
============

On OSX, two different types of Python Builds exist: a regular build and a
framework build. In order to interact correctly with OSX through the native
GUI frameworks you need a framework build of Python.
At the time of writing the ``macosx`` and ``WXAgg`` backends require a
framework build to function correctly. This can result in issues for
a python installation not build as a framework and may also happen in
virtual envs and when using (Ana)Conda.
From Matplotlib 1.5 onwards the ``macosx`` backend
checks that a framework build is available and fails if a non framework
build is found. WX has a similar check build in.

Without this check a partially functional figure is created.
Among the issues with it is that it is produced in the background and
cannot be put in front of any other window. Several solutions and work
arounds exist see below.

Short version
=============

VirtualEnv
----------

If you are on Python 3, use
`venv <https://docs.python.org/3/library/venv.html>`_
instead of `virtualenv <https://virtualenv.pypa.io/en/latest/>`_::

    python -m venv my-virtualenv
    source my-virtualenv/bin/activate

Otherwise you will need one of the workarounds below.

Pyenv
-----

If you are using pyenv and virtualenv you can enable your python version to be installed as a framework::

    PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install x.x.x

Conda
-----

The default python provided in (Ana)Conda is not a framework
build. However, the Conda developers have made it easy to install
a framework build in both the main environment and in Conda envs.
To use this install python.app ``conda install python.app`` and
use ``pythonw`` rather than ``python``


Long version
============

Unfortunately virtualenv creates a non
framework build even if created from a framework build of Python.
As documented above you can use venv as an alternative on Python 3.

The issue has been reported on the virtualenv bug tracker `here
<https://github.com/pypa/virtualenv/issues/54>`__ and `here
<https://github.com/pypa/virtualenv/issues/609>`__

Until this is fixed, one of the following workarounds can be used:

``PYTHONHOME`` Function
-----------------------

The best known work around is to use the non
virtualenv python along with the PYTHONHOME environment variable.
This can be done by defining a function in your ``.bashrc`` using

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

With this in place you can run ``frameworkpython`` to get an interactive
framework build within the virtualenv. To run a script you can do
``frameworkpython test.py`` where ``test.py`` is a script that requires a
framework build. To run an interactive ``IPython`` session with the framework
build within the virtual environment you can do ``frameworkpython -m IPython``

``PYTHONHOME`` and Jupyter
^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach can be followed even if using `Jupyter <https://jupyter.org/>`_ 
notebooks: you just need to setup a kernel with the suitable ``PYTHONHOME`` 
definition. The  `jupyter-virtualenv-osx  <https://github.com/mapio/jupyter-virtualenv-osx>`_ 
script automates the creation of such a kernel.


``PYTHONHOME`` Script
^^^^^^^^^^^^^^^^^^^^^

An alternative work around borrowed from the `WX wiki
<https://wiki.wxpython.org/wxPythonVirtualenvOnMac>`_, is to use the non
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
  ENV=`$PYTHON -c "import os; print(os.path.abspath(os.path.join(os.path.dirname(\"$0\"), '..')))"`

  # now run Python with the virtualenv set as Python's HOME
  export PYTHONHOME=$ENV
  exec $PYTHON "$@"

With this in place you can run ``frameworkpython`` as above but will need to add this script
to every virtualenv

PythonW Compiler
^^^^^^^^^^^^^^^^

In addition
`virtualenv-pythonw-osx <https://github.com/gldnspud/virtualenv-pythonw-osx>`_
provides an alternative workaround which may be used to solve the issue.
