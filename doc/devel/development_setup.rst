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

Installing additional dependencies for development
==================================================
See :ref:`development-dependencies`.
