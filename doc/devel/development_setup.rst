.. _installing_for_devs:

=====================================
Setting up Matplotlib for development
=====================================

To set up Matplotlib for development follow these steps:

.. contents::
   :local:

Retrieve the latest version of the code
=======================================

Matplotlib is hosted at https://github.com/matplotlib/matplotlib.git.

You can retrieve the latest sources with the command (see
:ref:`set-up-fork` for more details)

.. tab-set::

   .. tab-item:: https

      .. code-block:: bash

         git clone https://github.com/matplotlib/matplotlib.git

   .. tab-item:: ssh

      .. code-block:: bash

         git clone git@github.com:matplotlib/matplotlib.git

      This requires you to setup an `SSH key`_ in advance, but saves you from
      typing your password at every connection.

      .. _SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

This will place the sources in a directory :file:`matplotlib` below your
current working directory. Change into this directory::

    cd matplotlib


.. _dev-environment:

Create a dedicated environment
==============================
You should set up a dedicated environment to decouple your Matplotlib
development from other Python and Matplotlib installations on your system.

The simplest way to do this is to use either Python's virtual environment
`venv`_ or `conda`_.

.. _venv: https://docs.python.org/3/library/venv.html
.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

.. tab-set::

   .. tab-item:: venv environment

      Create a new `venv`_ environment with ::

        python -m venv <file folder location>

      and activate it with one of the following ::

        source <file folder location>/bin/activate  # Linux/macOS
        <file folder location>\Scripts\activate.bat  # Windows cmd.exe
        <file folder location>\Scripts\Activate.ps1  # Windows PowerShell

   .. tab-item:: conda environment

      Create a new `conda`_ environment with ::

        conda env create -f environment.yml

      You can use ``mamba`` instead of ``conda`` in the above command if
      you have `mamba`_ installed.

      .. _mamba: https://mamba.readthedocs.io/en/latest/

      Activate the environment using ::

        conda activate mpl-dev

Remember to activate the environment whenever you start working on Matplotlib.

Install Matplotlib in editable mode
===================================
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

Install additional development dependencies
===========================================
See :ref:`development-dependencies`.

Install pre-commit hooks (optional)
===================================
`pre-commit <https://pre-commit.com/>`_ hooks automatically check flake8 and
other style issues when you run ``git commit``. The hooks are defined in the
top level ``.pre-commit-config.yaml`` file. To install the hooks ::

    python -m pip install pre-commit
    pre-commit install

The hooks can also be run manually. All the hooks can be run, in order as
listed in ``.pre-commit-config.yaml``, against the full codebase with ::

    pre-commit run --all-files

To run a particular hook manually, run ``pre-commit run`` with the hook id ::

    pre-commit run <hook id> --all-files
