
.. _git_workflows:

Git workflows
=============

When reviewing contributions, we sometime refer to this page to offer more detailed git workflows.

.. _git_rebase:

Rebase
^^^^^^

If you want to rebase, the first thing to do is to squash all your commits into one, which will make the job easier.
Make sure you are in the PR branch, then to rebase do::

$ git rebase --interactive upstream/main


Git opens the last commits you made in our terminal editor (often it's vim) and you need to follow the instructions in
the file. Basically replace 'pick' by 'fixup' (or simply 'f') in all but the first commit (exit vim using `:wq` to
write/save and quite). Then update your main branch from upstream, change back to the PR branch and do::

$ git rebase main


and if there are problems, do ``$ git status`` to see which files need fixing, then edit the files to fix up any
conflicts (sections marked by "<<<") . When you are done with that::

$ git add <the fixed files>
$ git rebase --continue
$ git push --force-with-lease origin HEAD

If you have any problems, feel free to ask questions.

PS.
If at any point anything goes wrong, and you don't know what to do, just do::

$ git rebase --abort

and everything will go back to the way it was in the before ``$ git rebase`` times, and you can come back to your PR,
or to `gitter`_, and ask us for help, or that we do the rebase for you 😉.

.. _gitter: https://gitter.im/matplotlib/matplotlib
