.. _environment-variables:

*********************
Environment Variables
*********************

.. envvar:: HOME

  The user's home directory. On linux, :envvar:`~ <HOME>` is shorthand for :envvar:`HOME`.

.. envvar:: PATH

  The list of directories searched to find executable programs

.. envvar:: PYTHONPATH

  The list of directories that is added to Python's standard search list when
  importing packages and modules


Setting Environment Variables in Linux
======================================

To list the current value of :envvar:`PYTHONPATH`, which may be empty, try::

  echo $PYTHONPATH

The procedure for setting environment variables in depends on what your default
shell is. BASH seems to be the most common, but CSH is also common. You
should be able to determine which by running at the command prompt::

  echo $SHELL
  

BASH/KSH
--------

To create a new environment variable::

  export PYTHONPATH=~/Python

To prepend to an existing environment variable::

  export PATH=~/bin:${PATH}

The search order may be important to you, do you want `~/bin` to be searched
first or last? To append to an existing environment variable::

  export PATH=${PATH}:~/bin

To make your changes available in the future, add the commands to your
`~/.bashrc` file.


CSH/TCSH
--------

To create a new environment variable::

  setenv PYTHONPATH ~/Python

To prepend to an existing environment variable::

  setenv PATH ~/bin:${PATH}

The search order may be important to you, do you want `~/bin` to be searched
first or last? To append to an existing environment variable::

  setenv PATH ${PATH}:~/bin

To make your changes available in the future, add the commands to your
`~/.cshrc` file.


Setting Environment Variables in Windows
========================================

Go to the Windows start menu, select `Control Panel`, then select the `System`
icon, click the `Advanced` tab, and select the `Environment Variables`
button. You can edit or add to the `user variables`.