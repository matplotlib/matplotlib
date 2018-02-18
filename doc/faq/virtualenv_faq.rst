.. _virtualenv-faq:

***********************************************
Working with Matplotlib in Virtual environments
***********************************************

When running Matplotlib in a `virtual environment
<https://virtualenv.pypa.io/en/latest/>`_ you may discover a few issues.
Matplotlib itself has no issue with virtual environments.  However, some of
the external GUI frameworks that Matplotlib uses for interactive figures may
be tricky to install in a virtual environment.  Everything below assumes some
familiarity with the Matplotlib backends as found in :ref:`What is a backend?
<what-is-a-backend>`.

If you only use the IPython and Jupyter Notebook's ``inline`` and ``notebook``
backends, or non-interactive backends, you should not have any issues and can
ignore everything below.

Likewise, the ``Tk`` framework (``TkAgg`` backend) does not require any
external dependencies and is normally always available.  On certain Linux
distributions, a package named ``python-tk`` (or similar) needs to be
installed.

Otherwise, the situation (at the time of writing) is as follows:

============= ========================== =================================
GUI framework pip-installable?           conda or conda-forge-installable?
============= ========================== =================================
PyQt5         on Python>=3.5             yes
------------- -------------------------- ---------------------------------
PyQt4         PySide: on Windows and OSX yes
------------- -------------------------- ---------------------------------
PyGObject     no                         on Linux
------------- -------------------------- ---------------------------------
PyGTK         no                         no
------------- -------------------------- ---------------------------------
wxPython      yes [#]_                   yes
============= ========================== =================================

.. [#] OSX and Windows wheels available on PyPI.  Linux wheels available but
       not on PyPI, see https://wxpython.org/pages/downloads/.

In other cases, you need to install the package in the global (system)
site-packages, and somehow make it available from within the virtual
environment.  This can be achieved by any of the following methods (in all
cases, the system-wide Python and the virtualenv Python must be of the same
version):

- Using ``virtualenv``\'s ``--system-site-packages`` option when creating
  an environment adds all system-wide packages to the virtual environment.
  However, this breaks the isolation between the virtual environment and the
  system install.  Among other issues it results in hard to debug problems
  with system packages shadowing the environment packages.  If you use
  `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/>`_, this can be
  toggled with the ``toggleglobalsitepackages`` command.

- `vext <https://pypi.python.org/pypi/vext>`_ allows controlled access
  from within the virtualenv to specific system-wide packages without the
  overall shadowing issue.  A specific package needs to be installed for each
  framework, e.g. `vext.pyqt5 <https://pypi.python.org/pypi/vext.pyqt5>`_, etc.
  It is recommended to use ``vext>=0.7.0`` as earlier versions misconfigure the
  logging system.

If you are using Matplotlib on OSX, you may also want to consider the
:ref:`OSX framework FAQ <osxframework-faq>`.
