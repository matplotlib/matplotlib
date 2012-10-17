.. _environment-variables:

*********************
Environment Variables
*********************

.. contents::
   :backlinks: none


.. envvar:: HOME

  The user's home directory. On linux, :envvar:`~ <HOME>` is shorthand for :envvar:`HOME`.

.. envvar:: PATH

  The list of directories searched to find executable programs

.. envvar:: PYTHONPATH

  The list of directories that is added to Python's standard search list when
  importing packages and modules

.. envvar:: MPLCONFIGDIR

  This is the directory used to store user customizations to matplotlib, as
  well as some caches to improve performance. If :envvar:`MPLCONFIGDIR` is not
  defined, :file:`{HOME}/.matplotlib` is used if it is writable.
  Otherwise, the python standard library :func:`tempfile.gettmpdir` is
  used to find a base directory in which the :file:`matplotlib`
  subdirectory is created.

.. _setting-linux-osx-environment-variables:

Setting environment variables in Linux and OS-X
===============================================

To list the current value of :envvar:`PYTHONPATH`, which may be empty, try::

  echo $PYTHONPATH

The procedure for setting environment variables in depends on what your default
shell is. :program:`BASH` seems to be the most common, but :program:`CSH` is
also common. You should be able to determine which by running at the command
prompt::

  echo $SHELL


BASH/KSH
--------

To create a new environment variable::

  export PYTHONPATH=~/Python

To prepend to an existing environment variable::

  export PATH=~/bin:${PATH}

The search order may be important to you, do you want :file:`~/bin` to
be searched first or last? To append to an existing environment
variable::

  export PATH=${PATH}:~/bin

To make your changes available in the future, add the commands to your
:file:`~/.bashrc` file.


CSH/TCSH
--------

To create a new environment variable::

  setenv PYTHONPATH ~/Python

To prepend to an existing environment variable::

  setenv PATH ~/bin:${PATH}

The search order may be important to you, do you want :file:`~/bin` to be searched
first or last? To append to an existing environment variable::

  setenv PATH ${PATH}:~/bin

To make your changes available in the future, add the commands to your
:file:`~/.cshrc` file.

.. _setting-windows-environment-variables:

Setting environment variables in windows
========================================

Open the :program:`Control Panel` (:menuselection:`Start --> Control Panel`),
start the :program:`System` program. Click the :guilabel:`Advanced` tab
and select the :guilabel:`Environment Variables` button. You can edit or add to
the :guilabel:`User Variables`.
