.. _development-workflow:

=========================
Git Development workflow
=========================

You've discovered a bug or something else you want to change
in matplotlib_ .. |emdash| excellent!

You've worked out a way to fix it |emdash| even better!

You want to tell us about it |emdash| best of all!

The easiest way to contribute to matplotlib_ is through github_.  If
for some reason you don't want to use github, see
:ref:`making-patches` for instructions on how to email patches to the
mailing list.

You already have your own forked copy of the matplotlib_ repository, by
following :ref:`forking`, :ref:`set-up-fork`, and you have configured
git_ by following :ref:`configure-git`.

Workflow summary
================

* Keep your ``master`` branch clean of edits that have not been merged
  to the main matplotlib_ development repo.  Your ``master`` then will follow
  the main matplotlib_ repository.
* Start a new *feature branch* for each set of edits that you do.
* Do not merge the ``master`` branch or maintenance tracking branches
  into your feature branch.  If you need to include commits from upstream
  branches (either to pick up a bug fix or to resolve a conflict) please
  *rebase* your branch on the upstream branch.
* Ask for review!

This way of working really helps to keep work well organized, and in
keeping history as clear as possible.

See |emdash| for example |emdash| `linux git workflow`_.

Making a new feature branch
===========================

::

   git checkout -b my-new-feature master

This will create and immediately check out a feature branch based on
``master``.  To create a feature branch based on a maintenance branch,
use::

   git fetch origin
   git checkout -b my-new-feature origin/v1.0.x

Generally, you will want to keep this also on your public GitHub_ fork
of matplotlib_.  To do this, you `git push`_ this new branch up to your GitHub_
repo.  Generally (if you followed the instructions in these pages, and
by default), git will have a link to your GitHub_ repo, called
``origin``.  You push up to your own repo on GitHub_ with::

   git push origin my-new-feature

You will need to use this exact command, rather than simply ``git
push`` every time you want to push changes on your feature branch to
your GitHub_ repo.  However, in git >1.7 you can set up a link by
using the ``--set-upstream`` option::

   git push --set-upstream origin my-new-feature

and then next time you need to push changes to your branch a simple
``git push`` will suffice. Note that ``git push`` pushes out all
branches that are linked to a remote branch.

The editing workflow
====================

Overview
--------

::

   # hack hack
   git add my_new_file
   git commit -am 'NF - some message'
   git push

In more detail
--------------

#. Make some changes
#. See which files have changed with ``git status`` (see `git status`_).
   You'll see a listing like this one::

     # On branch ny-new-feature
     # Changed but not updated:
     #   (use "git add <file>..." to update what will be committed)
     #   (use "git checkout -- <file>..." to discard changes in working directory)
     #
     #  modified:   README
     #
     # Untracked files:
     #   (use "git add <file>..." to include in what will be committed)
     #
     #  INSTALL
     no changes added to commit (use "git add" and/or "git commit -a")

#. Check what the actual changes are with ``git diff`` (`git diff`_).
#. Add any new files to version control ``git add new_file_name`` (see
   `git add`_).
#. To commit all modified files into the local copy of your repo,, do
   ``git commit -am 'A commit message'``.  Note the ``-am`` options to
   ``commit``. The ``m`` flag just signals that you're going to type a
   message on the command line.  The ``a`` flag |emdash| you can just take on
   faith |emdash| or see `why the -a flag?`_ |emdash| and the helpful use-case
   description in the `tangled working copy problem`_. The `git commit`_ manual
   page might also be useful.
#. To push the changes up to your forked repo on GitHub_, do a ``git
   push`` (see `git push`).

Asking for code review |emdash| open a Pull Request (PR)
========================================================

It's a good idea to consult the :ref:`pull-request-checklist` to make
sure your pull request is ready for merging.


#. Go to your repo URL |emdash| e.g.,
   ``http://github.com/your-user-name/matplotlib``.

#. Select your feature branch from the drop down menu:

#. Click on the green button:

#. Make sure that you are requesting a pull against the correct branch

#. Enter a PR heading and description (if there is only one commit in
   the PR github will automatically fill these fields for you).  If
   this PR is addressing a specific issue, please reference it by number
   (ex #1325) which github will automatically make into links.

#. Click 'Create Pull Request' button!

#. Discussion of the change will take place in the pull request
   thread.


Rebasing a Pull Request (PR)
============================

When working on a PR, changes may occur in the parent branch (usually master).
This can lead to conflict with changes in your branch. The conflicts can be
trivial: for example both the parent branch and your branch add an entry to
the top of `CHANGELOG`.  Git can not unambiguously tell what to do with both
changes (should one go above the other? if so, which order? should it try to
merge them?) so it declares the branches can not be merged
cleanly. GitHub can only automatically merge PR without conflicts, so you will
need to manually 'rebase'. This is the process of updating your branch with
upstream changes, and resolving conflicts.

In git, rebasing is a mild form of re-writing history: it effectively forwards
all your commits to the updated upstream commit. For a much more detailed
explanation (with pictures!) see `this nice write up
<https://git-scm.com/book/en/Git-Branching-Rebasing>`_.  The NumPy team has also
`documented how to do this
<http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#rebasing-on-master>`_.
In general, re-writing history, particularly published history, is considered
bad practice, but in this case it is very useful.

The following example assumes that the remote of _your_ GitHub
repository is called `origin` and the remote of the official
repository is called `matplotlib`.

The first step is to make sure that your local copy of the upstream repository is
up-to-date::

     $ git fetch matplotlib

This updates your local copy of the repository, but does not change any files
in your working copy.  Next, switch to the branch that you want to update::

     $ git checkout backend_plt_refactor

You are now ready to start the rebase of your branch onto the target
parent branch, in this case `matplotlib/master` ::

     $ git rebase matplotlib/master

and git will then give a bunch of feed back::

     First, rewinding head to replay your work on top of it...
     Applying: first steps to extract FigureManager* and friends from pyplot
     Applying: split backend_qt4 into two parts, with and without Gcf
     ...
     Applying: pep8 clean up on backend_gtk3.py
     Using index info to reconstruct a base tree...
     M       lib/matplotlib/backends/backend_gtk3.py
     Falling back to patching base and 3-way merge...
     Auto-merging lib/matplotlib/backends/backend_gtk3.py
     CONFLICT (content): Merge conflict in lib/matplotlib/backends/backend_gtk3.py
     Failed to merge in the changes.
     Patch failed at 0013 pep8 clean up on backend_gtk3.py
     The copy of the patch that failed is found in:
        /home/tcaswell/other_source/matplotlib/.git/rebase-apply/patch

     When you have resolved this problem, run "git rebase --continue".
     If you prefer to skip this patch, run "git rebase --skip" instead.
     To check out the original branch and stop rebasing, run "git rebase --abort".

We see that a number of commits could be cleanly applied to the tip of
`matplotlib/master`.  However, git may eventually hit a commit that
had conflicts: in the example above, this happens in the file
`lib/matplotlib/backends/backend_gtk3.py`).  For more verbose
information run ::

     $ git status

     You are currently rebasing branch 'backend_plt_refactor' on 'e6f8993'.
       (fix conflicts and then run "git rebase --continue")
       (use "git rebase --skip" to skip this patch)
       (use "git rebase --abort" to check out the original branch)

     Unmerged paths:
       (use "git reset HEAD <file>..." to unstage)
       (use "git add <file>..." to mark resolution)

             both modified:      lib/matplotlib/backends/backend_gtk3.py

     no changes added to commit (use "git add" and/or "git commit -a")

This tells you exactly where the conflict (caused by the target branch
and your commits modifying the same lines of code) is and provides
some advice on how to proceed.  Opening up the file in question, you
will see blocks that look something like this::

     <<<<<<< HEAD
     =======
             self.__dict__.clear()   # Is this needed? Other backends don't have it.
     >>>>>>> pep8 clean up on backend_gtk3.py

The block of code between `<<<<<<<` and `=======` is the code on the
target branch (in this case nothing) and the code between `=======`
and `>>>>>>>` is the code in the commit you are trying to rebase.  The
rest of the code is either the same or the diff can be unambiguously
applied.  You need to determine how to resolve the conflict (in this
case, the code on HEAD is correct).  Once you have resolved all the
conflicts, `add` the file to the index::

     $ git add lib/matplotlib/backends/backend_gtk3.py

Repeat this for all of the files that have conflicts.  When you are done with
that you can check the status::

     $ git status
     rebase in progress; onto e6f8993
     You are currently rebasing branch 'backend_plt_refactor' on 'e6f8993'.
       (all conflicts fixed: run "git rebase --continue")

     Changes to be committed:
       (use "git reset HEAD <file>..." to unstage)

             modified:   lib/matplotlib/backends/backend_gtk3.py

which shows us that we have resolved all of the conflicts with this
commit and can continue::

     $ git rebase --continue

You now iterate the until you have made it through all of the commits
which have conflicts.  Once you have successfully rebased your branch,
be sure to re-run the tests to make sure everything is still working
properly.

Your branch is now rebased, however, because of the way git
determines the hash of each commit, it now shares no commits with your
old branch published on GitHub so you can not push to that branch as
you would when simply adding commits.  In order to publish your newly
rebased (and tested!) branch you need to use the `--force` flag::

    $ git push --force origin

which will _replace_ all of the commits under your branch on GitHub
with the new versions of the commit.

Congratulations, you have rebased your branch!

Staying up to date with changes in the central repository
=========================================================

This updates your working copy from the upstream `matplotlib GitHub`_
repo.

Overview
--------

::

   # go to your master branch
   git checkout master
   # pull changes from github
   git fetch matplotlib
   # merge from matplotlib
   git merge --ff-only matplotlib/master

In detail
---------

We suggest that you do this only for your ``master`` branch, and leave
your 'feature' branches unmerged, to keep their history as clean as
possible.  This makes code review easier::

   git checkout master

Make sure you have done :ref:`linking-to-upstream`.

Merge the upstream code into your current development by first pulling
the upstream repo to a copy on your local machine::

   git fetch upstream

then merging into your current branch::

   git merge --ff-only upstream/master

The ``--ff-only`` option guarantees that if you have mistakenly
committed code on your ``master`` branch, the merge fails at this point.
If you were to merge ``upstream/master`` to your ``master``, you
would start to diverge from the upstream. If this command fails, see
the section on accidents_.

The letters 'ff' in ``--ff-only`` mean 'fast forward', which is a
special case of merge where git can simply update your branch to point
to the other branch and not do any actual merging of files. For
``master`` and other integration branches this is exactly what you
want.

Other integration branches
--------------------------

Some people like to keep separate local branches corresponding to the
maintenance branches on GitHub. At the time of this writing, ``v1.0.x``
is the active maintenance branch. If you have such a local branch,
treat is just as ``master``: don't commit on it, and before starting
new branches off of it, update it from upstream::

   git checkout v1.0.x
   git fetch upstream
   git merge --ff-only upstream/v1.0.x

But you don't necessarily have to have such a branch. Instead, if you
are preparing a bugfix that applies to the maintenance branch, fetch
from upstream and base your bugfix on the remote branch::

   git fetch upstream
   git checkout -b my-bug-fix upstream/v1.0.x

.. _accidents:

Recovering from accidental commits on master
--------------------------------------------

If you have accidentally committed changes on ``master`` and
``git merge --ff-only`` fails, don't panic! First find out how much
you have diverged::

   git diff upstream/master...master

If you find that you want simply to get rid of the changes, reset
your ``master`` branch to the upstream version::

   git reset --hard upstream/master

As you might surmise from the words 'reset' and 'hard', this command
actually causes your changes to the current branch to be lost, so
think twice.

If, on the other hand, you find that you want to preserve the changes,
create a feature branch for them::

   git checkout -b my-important-changes

Now ``my-important-changes`` points to the branch that has your
changes, and you can safely reset ``master`` as above |emdash| but
make sure to reset the correct branch::

   git checkout master
   git reset --hard upstream/master


Deleting a branch on GitHub_
============================

::

   git checkout master
   # delete branch locally
   git branch -D my-unwanted-branch
   # delete branch on GitHub
   git push origin :my-unwanted-branch

Note the colon ``:`` before ``my-unwanted-branch``.  See also:
https://help.github.com/articles/pushing-to-a-remote/#deleting-a-remote-branch-or-tag

Exploring your repository
=========================

To see a graphical representation of the repository branches and
commits::

   gitk --all

To see a linear list of commits for this branch::

   git log

You can also look at the `network graph visualizer`_ for your GitHub_
repo.

.. include:: links.inc
