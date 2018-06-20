.. _osxframework-faq:

******************************
Working with Matplotlib on OSX
******************************

.. contents::
   :backlinks: none

.. _osxframework_introduction:

Introduction
============

On OSX, two different types of Python builds exist: a regular build and a
framework build.  In order to interact correctly with OSX through the native
GUI frameworks, you need a framework build of Python.  At the time of writing
the ``macosx`` and ``WXAgg`` backends require a framework build to function
correctly.  This can result in issues for a Python installation not build as a
framework and may also happen in virtual envs and when using (Ana)conda.  From
Matplotlib 1.5 onwards, both backends check that a framework build is available
and fail if a non framework build is found.  (Without this check a partially
functional figure is created.  In particular, it is produced in the background
and cannot be put in front of any other window.)

virtualenv
----------

In a virtualenv_, a non-framework build is used even when the environment is
created from a framework build (`virtualenv bug #54`_, `virtualenv bug #609`_).

The solution is to not use virtualenv, but instead the stdlib's venv_, which
provides similar functionality but without exhibiting this issue.

If you absolutely need to use virtualenv rather than venv, then from within
the environment you can set the ``PYTHONHOME`` environment variable to
``$VIRTUAL_ENV``, then invoke Python using the full path to the framework build
(typically ``/usr/local/bin/python``).

.. _virtualenv: https://virtualenv.pypa.io/
.. _virtualenv bug #54: https://github.com/pypa/virtualenv/issues/54
.. _virtualenv bug #609: https://github.com/pypa/virtualenv/issues/609
.. _venv: https://docs.python.org/3/library/venv.html

conda
-----

The default python provided in (Ana)conda is not a framework build.  However,
a framework build can easily be installed, both in the main environment and
in conda envs: install python.app (``conda install python.app``) and use
``pythonw`` rather than ``python``
