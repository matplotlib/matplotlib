.. highlight:: bash

.. _configure-git:

===============
 Configure git
===============

.. _git-config-basic:

Overview
========

Your personal git configurations are saved in the ``.gitconfig`` file in
your home directory.

Here is an example ``.gitconfig`` file:

.. code-block:: none

  [user]
          name = Your Name
          email = you@yourdomain.example.com

  [alias]
          ci = commit -a
          co = checkout
          st = status
          stat = status
          br = branch
          wdiff = diff --color-words

  [core]
          editor = vim

  [merge]
          summary = true

You can edit this file directly or you can use the ``git config --global``
command::

  git config --global user.name "Your Name"
  git config --global user.email you@yourdomain.example.com
  git config --global alias.ci "commit -a"
  git config --global alias.co checkout
  git config --global alias.st "status -a"
  git config --global alias.stat "status -a"
  git config --global alias.br branch
  git config --global alias.wdiff "diff --color-words"
  git config --global core.editor vim
  git config --global merge.summary true

To set up on another computer, you can copy your ``~/.gitconfig`` file,
or run the commands above.

In detail
=========

user.name and user.email
------------------------

It is good practice to tell git_ who you are, for labeling any changes
you make to the code.  The simplest way to do this is from the command
line::

  git config --global user.name "Your Name"
  git config --global user.email you@yourdomain.example.com

This will write the settings into your git configuration file,  which
should now contain a user section with your name and email:

.. code-block:: none

  [user]
        name = Your Name
        email = you@yourdomain.example.com

Of course you'll need to replace ``Your Name`` and ``you@yourdomain.example.com``
with your actual name and email address.

Aliases
-------

You might well benefit from some aliases to common commands.

For example, you might well want to be able to shorten ``git checkout``
to ``git co``.  Or you may want to alias ``git diff --color-words``
(which gives a nicely formatted output of the diff) to ``git wdiff``

The following ``git config --global`` commands::

  git config --global alias.ci "commit -a"
  git config --global alias.co checkout
  git config --global alias.st "status -a"
  git config --global alias.stat "status -a"
  git config --global alias.br branch
  git config --global alias.wdiff "diff --color-words"

will create an ``alias`` section in your ``.gitconfig`` file with contents
like this:

.. code-block:: none

  [alias]
          ci = commit -a
          co = checkout
          st = status -a
          stat = status -a
          br = branch
          wdiff = diff --color-words

Editor
------

You may also want to make sure that your editor of choice is used ::

  git config --global core.editor vim

Merging
-------

To enforce summaries when doing merges (``~/.gitconfig`` file again):

.. code-block:: none

   [merge]
      log = true

Or from the command line::

  git config --global merge.log true

.. _fancy-log:

Fancy log output
----------------

This is a very nice alias to get a fancy log output; it should go in the
``alias`` section of your ``.gitconfig`` file:

.. code-block:: none

    lg = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)[%an]%Creset' --abbrev-commit --date=relative

You use the alias with::

    git lg

and it gives graph / text output something like this (but with color!):

.. code-block:: none

    * 6d8e1ee - (HEAD, origin/my-fancy-feature, my-fancy-feature) NF - a fancy file (45 minutes ago) [Matthew Brett]
    *   d304a73 - (origin/placeholder, placeholder) Merge pull request #48 from hhuuggoo/master (2 weeks ago) [Jonathan Terhorst]
    |\  
    | * 4aff2a8 - fixed bug 35, and added a test in test_bugfixes (2 weeks ago) [Hugo]
    |/  
    * a7ff2e5 - Added notes on discussion/proposal made during Data Array Summit. (2 weeks ago) [Corran Webster]
    * 68f6752 - Initial implimentation of AxisIndexer - uses 'index_by' which needs to be changed to a call on an Axes object - this is all very sketchy right now. (2 weeks ago) [Corr
    *   376adbd - Merge pull request #46 from terhorst/master (2 weeks ago) [Jonathan Terhorst]
    |\  
    | * b605216 - updated joshu example to current api (3 weeks ago) [Jonathan Terhorst]
    | * 2e991e8 - add testing for outer ufunc (3 weeks ago) [Jonathan Terhorst]
    | * 7beda5a - prevent axis from throwing an exception if testing equality with non-axis object (3 weeks ago) [Jonathan Terhorst]
    | * 65af65e - convert unit testing code to assertions (3 weeks ago) [Jonathan Terhorst]
    | *   956fbab - Merge remote-tracking branch 'upstream/master' (3 weeks ago) [Jonathan Terhorst]
    | |\  
    | |/

Thanks to Yury V. Zaytsev for posting it.

.. include:: links.inc
