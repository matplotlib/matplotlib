.. _installing_for_devs:

=====================================
Setting up Matplotlib for development
=====================================

.. _dev-environment:

Creating a dedicated environment
================================
You should set up a dedicated environment to decouple your Matplotlib
development from other Python and Matplotlib installations on your system.
Here we use python's virtual environment `venv`_, but you may also use others
such as conda.

.. _venv: https://docs.python.org/3/library/venv.html

A new environment can be set up with ::

   python -m venv <file folder location>

and activated with one of the following::

   source <file folder location>/bin/activate  # Linux/macOS
   <file folder location>\Scripts\activate.bat  # Windows cmd.exe
   <file folder location>\Scripts\Activate.ps1  # Windows PowerShell

Whenever you plan to work on Matplotlib, remember to activate the development
environment in your shell.

Retrieving the latest version of the code
=========================================

Matplotlib is hosted at https://github.com/matplotlib/matplotlib.git.

You can retrieve the latest sources with the command (see
:ref:`set-up-fork` for more details)::

    git clone https://github.com/matplotlib/matplotlib.git

This will place the sources in a directory :file:`matplotlib` below your
current working directory.

If you have the proper privileges, you can use ``git@`` instead of
``https://``, which works through the ssh protocol and might be easier to use
if you are using 2-factor authentication.

Installing Matplotlib in editable mode
======================================
Install Matplotlib in editable mode from the :file:`matplotlib` directory
using the command ::

    python -m pip install -ve .

The 'editable/develop mode', builds everything and places links in your Python
environment so that Python will be able to import Matplotlib from your
development source directory.  This allows you to import your modified version
of Matplotlib without re-installing after every change. Note that this is only
true for ``*.py`` files.  If you change the C-extension source (which might
also happen if you change branches) you will have to re-run
``python -m pip install -ve .``

.. _test-dependencies:

Additional dependencies for testing
===================================
This section lists the additional software required for
:ref:`running the tests <testing>`.

Required:

- pytest_ (>=3.6)
- Ghostscript_ (>= 9.0, to render PDF files)
- Inkscape_ (to render SVG files)

Optional:

- pytest-cov_ (>=2.3.1) to collect coverage information
- pytest-flake8_ to test coding standards using flake8_
- pytest-timeout_ to limit runtime in case of stuck tests
- pytest-xdist_ to run tests in parallel

.. _pytest: http://doc.pytest.org/en/latest/
.. _Ghostscript: https://www.ghostscript.com/
.. _Inkscape: https://inkscape.org
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _pytest-flake8: https://pypi.org/project/pytest-flake8/
.. _pytest-xdist: https://pypi.org/project/pytest-xdist/
.. _pytest-timeout: https://pypi.org/project/pytest-timeout/
.. _flake8: https://pypi.org/project/flake8/


.. _doc-dependencies:

Additional dependencies for building documentation
==================================================

Python packages
---------------
The additional Python packages required to build the
:ref:`documentation <documenting-matplotlib>` are listed in
:file:`doc-requirements.txt` and can be installed using ::

    pip install -r requirements/doc/doc-requirements.txt

The content of :file:`doc-requirements.txt` is also shown below:

   .. include:: ../../requirements/doc/doc-requirements.txt
      :literal:

Additional external dependencies
--------------------------------
Required:

*  a minimal working LaTeX distribution
*  `Graphviz <http://www.graphviz.org/download>`_
*  the LaTeX packages *cm-super* and *dvipng*. If your OS bundles ``TexLive``,
   then often the "complete" version of the installer will automatically include
   these packages (e.g. "texlive-full" or "texlive-all").

Optional, but recommended:

*  `Inkscape <https://inkscape.org>`_
*  `optipng <http://optipng.sourceforge.net>`_
*  the font "Humor Sans" (aka the "XKCD" font), or the free alternative
   `Comic Neue <http://comicneue.com/>`_.
*  the font "Times New Roman"

.. note::

  The documentation will not build without LaTeX and Graphviz.  These are not
  Python packages and must be installed separately. The documentation can be
  built without Inkscape and optipng, but the build process will raise various
  warnings. If the build process warns that you are missing fonts, make sure
  your LaTeX distribution bundles cm-super or install it separately.