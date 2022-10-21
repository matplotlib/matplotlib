.. highlight:: bash

.. _set-up-fork:

==================
 Set up your fork
==================

First you follow the instructions for :ref:`forking`.

Overview
========

::

   git clone https://github.com/your-user-name/matplotlib.git
   cd matplotlib
   git remote add upstream https://github.com/matplotlib/matplotlib.git

In detail
=========

Clone your fork
---------------

#. Clone your fork to the local computer with ``git clone
   https://github.com/your-user-name/matplotlib.git``
#. Investigate.  Change directory to your new repo: ``cd matplotlib``. Then
   ``git branch -a`` to show you all branches.  You'll get something
   like:

   .. code-block:: none

      * main
      remotes/origin/main

   This tells you that you are currently on the ``main`` branch, and
   that you also have a ``remote`` connection to ``origin/main``.
   What remote repository is ``remote/origin``? Try ``git remote -v`` to
   see the URLs for the remote.  They will point to your github fork.

   Now you want to connect to the upstream `Matplotlib github`_ repository, so
   you can merge in changes from trunk.

.. _linking-to-upstream:

Linking your repository to the upstream repo
--------------------------------------------

::

   cd matplotlib
   git remote add upstream https://github.com/matplotlib/matplotlib.git

``upstream`` here is just the arbitrary name we're using to refer to the
main `Matplotlib`_ repository at `Matplotlib github`_.

Just for your own satisfaction, show yourself that you now have a new
'remote', with ``git remote -v show``, giving you something like:

.. code-block:: none

   upstream	https://github.com/matplotlib/matplotlib.git (fetch)
   upstream	https://github.com/matplotlib/matplotlib.git (push)
   origin	https://github.com/your-user-name/matplotlib.git (fetch)
   origin	https://github.com/your-user-name/matplotlib.git (push)

.. include:: links.inc
