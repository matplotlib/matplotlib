.. _installing_for_devs:

=====================================
Setting up Matplotlib for development
=====================================

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

.. _dev-environment:

Creating a dedicated environment
================================
You should set up a dedicated environment to decouple your Matplotlib
development from other Python and Matplotlib installations on your system.

Using virtual environments
--------------------------

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

Conda dev environment
---------------------
After you have cloned the repo change into the matplotlib directory.

A new conda environment can be set-up with::

    conda env create -f environment.yml

Note that if you have mamba installed you can replace conda with mamba in
the above command.

To activate your environment::

    conda activate mpl-dev

Finish the install by the following command::

    pip install -e .

Whenever you plan to work on Matplotlib, remember to ``conda activate mpl-dev``
in your shell.

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

Installing pre-commit hooks
===========================
You can optionally install `pre-commit <https://pre-commit.com/>`_ hooks.
These will automatically check flake8 and other style issues when you run
``git commit``. The hooks are defined in the top level
``.pre-commit-config.yaml`` file. To install the hooks ::

    python -m pip install pre-commit
    pre-commit install

The hooks can also be run manually. All the hooks can be run, in order as
listed in ``.pre-commit-config.yaml``, against the full codebase with ::

    pre-commit run --all-files

To run a particular hook manually, run ``pre-commit run`` with the hook id ::

    pre-commit run <hook id> --all-files

Installing additional dependencies for development
==================================================
See :ref:`development-dependencies`.
