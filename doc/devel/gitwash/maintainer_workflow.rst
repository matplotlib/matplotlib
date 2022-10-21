.. highlight:: bash

.. _maintainer-workflow:

###################
Maintainer workflow
###################

This page is for maintainers |emdash| those of us who merge our own or other
peoples' changes into the upstream repository.

Being as how you're a maintainer, you are completely on top of the basic stuff
in :ref:`development-workflow`.

The instructions in :ref:`linking-to-upstream` add a remote that has read-only
access to the upstream repo.  Being a maintainer, you've got read-write access.

It's good to have your upstream remote have a scary name, to remind you that
it's a read-write remote::

    git remote add upstream-rw git@github.com:matplotlib/matplotlib.git
    git fetch upstream-rw

*******************
Integrating changes
*******************

Let's say you have some changes that need to go into trunk
(``upstream-rw/main``).

The changes are in some branch that you are currently on.  For example, you are
looking at someone's changes like this::

    git remote add someone https://github.com/someone/matplotlib.git
    git fetch someone
    git branch cool-feature --track someone/cool-feature
    git checkout cool-feature

So now you are on the branch with the changes to be incorporated upstream.  The
rest of this section assumes you are on this branch.

A few commits
=============

If there are only a few commits, consider rebasing to upstream::

    # Fetch upstream changes
    git fetch upstream-rw
    # rebase
    git rebase upstream-rw/main

Remember that, if you do a rebase, and push that, you'll have to close any
github pull requests manually, because github will not be able to detect the
changes have already been merged.

A long series of commits
========================

If there are a longer series of related commits, consider a merge instead::

    git fetch upstream-rw
    git merge --no-ff upstream-rw/main

The merge will be detected by github, and should close any related pull requests
automatically.

Note the ``--no-ff`` above.  This forces git to make a merge commit, rather than
doing a fast-forward, so that these set of commits branch off trunk then rejoin
the main history with a merge, rather than appearing to have been made directly
on top of trunk.

Check the history
=================

Now, in either case, you should check that the history is sensible and you have
the right commits::

    git log --oneline --graph
    git log -p upstream-rw/main..

The first line above just shows the history in a compact way, with a text
representation of the history graph. The second line shows the log of commits
excluding those that can be reached from trunk (``upstream-rw/main``), and
including those that can be reached from current HEAD (implied with the ``..``
at the end). So, it shows the commits unique to this branch compared to trunk.
The ``-p`` option shows the diff for these commits in patch form.

Push to trunk
=============

::

    git push upstream-rw my-new-feature:main

This pushes the ``my-new-feature`` branch in this repository to the ``main``
branch in the ``upstream-rw`` repository.

.. include:: links.inc
