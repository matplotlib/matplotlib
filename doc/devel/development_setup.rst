.. _installing_for_devs:

=====================================
Setting up Matplotlib for development
=====================================

.. note::

    Setting up everything correctly and working with git can be challenging for
    new contributors. If you have difficulties, feel free to ask for help. We
    have a dedicated :ref:`contributor-incubator` channel on gitter for
    onboarding new contributors.

Setting up git and retrieving the latest version of the code
============================================================

Matplotlib is hosted on `GitHub <https://github.com/matplotlib/matplotlib>`_.
To contribute you will need to sign up for a `free GitHub account
<https://github.com/signup/free>`_.

GitHub uses Git for version control, which allows many people to work together
on the project. See the following links for more information:

- `GitHub help pages <https://help.github.com/>`_
- `NumPy documentation <https://numpy.org/doc/stable/dev/index.html>`_
- `pandas documentation <https://pandas.pydata.org/docs/development/contributing.html#working-with-the-code>`_

Forking
-------
Go to https://github.com/matplotlib/matplotlib and click the ``Fork`` button in
the top right corner to create your `own copy of the project <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-forks>`_.

Now clone your fork to your machine (`GitHub help: cloning <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`_):

.. code-block:: bash

    git clone https://github.com/your-user-name/matplotlib.git

This creates a directory :file:`matplotlib` under your current working
directory with the Matplotlib source code. Change into it:

.. code-block:: bash

    cd matplotlib

and connect to the official Matplotlib repository under the remote name
``upstream`` (`GitHub help: add remote <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork>`_):

.. code-block::

    git remote add upstream https://github.com/matplotlib/matplotlib.git

Instead of cloning using the ``https://`` protocol, you can use the ssh
protocol (``git clone git@github.com:your-user-name/matplotlib.git``). This
needs `additional configuration`_ but lets you connect to GitHub without
having to enter your username and password.

.. _additional configuration: https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh

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
