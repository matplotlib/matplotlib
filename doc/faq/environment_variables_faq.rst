.. _environment-variables:

*********************
Environment Variables
*********************

.. contents::
   :backlinks: none


.. envvar:: DISPLAY

  The server and screen on which to place windows. This is interpreted by GUI
  toolkits in a backend-specific manner, but generally refers to an `X.org
  display name
  <https://www.x.org/releases/X11R7.7/doc/man/man7/X.7.xhtml#heading5>`_.

.. envvar:: HOME

  The user's home directory. On Linux, :envvar:`~ <HOME>` is shorthand for :envvar:`HOME`.

.. envvar:: MPLBACKEND

  This optional variable can be set to choose the Matplotlib backend. See
  :ref:`what-is-a-backend`.

.. envvar:: MPLCONFIGDIR

  This is the directory used to store user customizations to
  Matplotlib, as well as some caches to improve performance. If
  :envvar:`MPLCONFIGDIR` is not defined, :file:`{HOME}/.config/matplotlib`
  and :file:`{HOME}/.cache/matplotlib` are used on Linux, and
  :file:`{HOME}/.matplotlib` on other platforms, if they are
  writable. Otherwise, the Python standard library's `tempfile.gettempdir` is
  used to find a base directory in which the :file:`matplotlib` subdirectory is
  created.

.. envvar:: PATH

  The list of directories searched to find executable programs.

.. envvar:: PYTHONPATH

  The list of directories that are added to Python's standard search list when
  importing packages and modules.

.. envvar:: QT_API

   The Python Qt wrapper to prefer when using Qt-based backends. See :ref:`the
   entry in the usage guide <QT_API-usage>` for more information.

.. _setting-linux-osx-environment-variables:

Setting environment variables in Linux and macOS
================================================

To list the current value of :envvar:`PYTHONPATH`, which may be empty, try::

  echo $PYTHONPATH

The procedure for setting environment variables in depends on what your default
shell is.  Common shells include :program:`bash` and :program:`csh`.  You
should be able to determine which by running at the command prompt::

  echo $SHELL

To create a new environment variable::

  export PYTHONPATH=~/Python  # bash/ksh
  setenv PYTHONPATH ~/Python  # csh/tcsh

To prepend to an existing environment variable::

  export PATH=~/bin:${PATH}  # bash/ksh
  setenv PATH ~/bin:${PATH}  # csh/tcsh

The search order may be important to you, do you want :file:`~/bin` to be
searched first or last?  To append to an existing environment variable::

  export PATH=${PATH}:~/bin  # bash/ksh
  setenv PATH ${PATH}:~/bin  # csh/tcsh

To make your changes available in the future, add the commands to your
:file:`~/.bashrc`/:file:`.cshrc` file.

.. _setting-windows-environment-variables:

Setting environment variables in Windows
========================================

Open the :program:`Control Panel` (:menuselection:`Start --> Control Panel`),
start the :program:`System` program. Click the :guilabel:`Advanced` tab
and select the :guilabel:`Environment Variables` button. You can edit or add to
the :guilabel:`User Variables`.
