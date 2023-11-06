.. redirect-from:: /devel/gitwash/configure_git
.. redirect-from:: /devel/gitwash/dot2_dot3
.. redirect-from:: /devel/gitwash/following_latest
.. redirect-from:: /devel/gitwash/forking_hell
.. redirect-from:: /devel/gitwash/git_development
.. redirect-from:: /devel/gitwash/git_install
.. redirect-from:: /devel/gitwash/git_intro
.. redirect-from:: /devel/gitwash/git_resources
.. redirect-from:: /devel/gitwash/patching
.. redirect-from:: /devel/gitwash/set_up_fork
.. redirect-from:: /devel/gitwash/index

.. _installing_for_devs:

=====================================
Setting up Matplotlib for development
=====================================

To set up Matplotlib for development follow these steps:

.. contents::
   :local:

Fork the Matplotlib repository
==============================

Matplotlib is hosted at https://github.com/matplotlib/matplotlib.git. If you
plan on solving issues or submitting pull requests to the main Matplotlib
repository, you should first *fork* this repository by visiting
https://github.com/matplotlib/matplotlib.git and clicking on the
``Fork`` :octicon:`repo-forked` button on the top right of the page. See
`the GitHub documentation <https://docs.github.com/get-started/quickstart/fork-a-repo>`__
for more details.

Retrieve the latest version of the code
=======================================

Now that your fork of the repository lives under your GitHub username, you can
retrieve the most recent version of the source code with one of the following
commands (replace ``<your-username>`` with your GitHub username):

.. tab-set::

   .. tab-item:: https

      .. code-block:: bash

         git clone https://github.com/<your-username>/matplotlib.git

   .. tab-item:: ssh

      .. code-block:: bash

         git clone git@github.com:<your-username>/matplotlib.git

      This requires you to setup an `SSH key`_ in advance, but saves you from
      typing your password at every connection.

      .. _SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh


This will place the sources in a directory :file:`matplotlib` below your
current working directory and set the remote name ``origin`` to point to your
fork. Change into this directory before continuing::

    cd matplotlib

Now set the remote name ``upstream`` to point to the Matplotlib main repository:

.. tab-set::

   .. tab-item:: https

      .. code-block:: bash

         git remote add upstream https://github.com/matplotlib/matplotlib.git

   .. tab-item:: ssh

      .. code-block:: bash

         git remote add upstream git@github.com:matplotlib/matplotlib.git

You can now use ``upstream`` to retrieve the most current snapshot of the source
code, as described in :ref:`development-workflow`.

.. dropdown:: Additional ``git`` and ``GitHub`` resources
   :color: info
   :open:

   For more information on ``git`` and ``GitHub``, see:

   * `Git documentation <https://git-scm.com/doc>`_
   * `GitHub-Contributing to a Project
     <https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>`_
   * `GitHub Skills <https://skills.github.com/>`_
   * :ref:`using-git`
   * :ref:`git-resources`
   * `Installing git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
   * `Managing remote repositories
     <https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories>`_
   * https://tacaswell.github.io/think-like-git.html
   * https://tom.preston-werner.com/2009/05/19/the-git-parable.html

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

      On some systems, you may need to type ``python3`` instead of ``python``.
      For a discussion of the technical reasons, see `PEP-394 <https://peps.python.org/pep-0394>`_.

      Install the Python dependencies with ::

        pip install -r requirements/dev/dev-requirements.txt

   .. tab-item:: conda environment

      Create a new `conda`_ environment and install the Python dependencies with ::

        conda env create -f environment.yml

      You can use ``mamba`` instead of ``conda`` in the above command if
      you have `mamba`_ installed.

      .. _mamba: https://mamba.readthedocs.io/en/latest/

      Activate the environment using ::

        conda activate mpl-dev

Remember to activate the environment whenever you start working on Matplotlib.

Install Dependencies
====================
Most Python dependencies will be installed when :ref:`setting up the environment <dev-environment>`
but non-Python dependencies like C++ compilers, LaTeX, and other system applications
must be installed separately. For a full list, see :ref:`dependencies`.

.. _development-install:

Install Matplotlib in editable mode
===================================

Install Matplotlib in editable mode from the :file:`matplotlib` directory using the
command ::

    python -m pip install --verbose --no-build-isolation --editable .[dev]

The 'editable/develop mode' builds everything and places links in your Python environment
so that Python will be able to import Matplotlib from your development source directory.
This allows you to import your modified version of Matplotlib without having to
re-install after changing a ``.py`` or compiled extension file.

When working on a branch that does not have Meson enabled, meaning it does not
have :ghpull:`26621` in its history (log), you will have to reinstall from source
each time you change any compiled extension code.

If the installation is not working, please consult the :ref:`troubleshooting guide <troubleshooting-faq>`.
If the guide does not offer a solution, please reach out via `chat <https://gitter.im/matplotlib/matplotlib>`_
or :ref:`open an issue <submitting-a-bug-report>`.


Build options
-------------
If you are working heavily with files that need to be compiled, you may want to
inspect the compilation log. This can be enabled by setting the environment
variable :envvar:`MESONPY_EDITABLE_VERBOSE` or by setting the ``editable-verbose``
config during installation ::

   python -m pip install --no-build-isolation --config-settings=editable-verbose=true --editable .

For more information on installation and other configuration options, see the
Meson Python :external+meson-python:ref:`editable installs guide <how-to-guides-editable-installs>`.

Verify the Installation
=======================

Run the following command to make sure you have correctly installed Matplotlib in
editable mode. The command should be run when the virtual environment is activated::

    python -c "import matplotlib; print(matplotlib.__file__)"

This command should return : ``<matplotlib_local_repo>\lib\matplotlib\__init__.py``

We encourage you to run tests and build docs to verify that the code installed correctly
and that the docs build cleanly, so that when you make code or document related changes
you are aware of the existing issues beforehand.

* Run test cases to verify installation :ref:`testing`
* Verify documentation build :ref:`documenting-matplotlib`

.. _pre-commit-hooks:

Install pre-commit hooks
========================
`pre-commit <https://pre-commit.com/>`_ hooks save time in the review process by
identifying issues with the code before a pull request is formally opened. Most
hooks can also aide in fixing the errors, and the checks should have
corresponding :ref:`development workflow <development-workflow>` and
:ref:`pull request <pr-guidelines>` guidelines. Hooks are configured in
`.pre-commit-config.yaml <https://github.com/matplotlib/matplotlib/blob/main/.pre-commit-config.yaml?>`_
and include checks for spelling and formatting, flake 8 conformity, accidentally
committed files, import order, and incorrect branching.

Install pre-commit hooks ::

    python -m pip install pre-commit
    pre-commit install

Hooks are run automatically after the ``git commit`` stage of the
:ref:`editing workflow<edit-flow>`. When a hook has found and fixed an error in a
file, that file must be *staged and committed* again.

Hooks can also be run manually. All the hooks can be run, in order as
listed in ``.pre-commit-config.yaml``, against the full codebase with ::

    pre-commit run --all-files

To run a particular hook manually, run ``pre-commit run`` with the hook id ::

    pre-commit run <hook id> --all-files
