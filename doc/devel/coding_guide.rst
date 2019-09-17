.. _reviewers-guide:

********************
Reviewers guideline
********************

.. _pull-request-checklist:

Pull request checklist
======================

Branch selection
----------------

Generally, all pull requests (*PRs*) should target the master branch. Other
branches are fed through automatic or manual backports. Directly targeting
other branches is only rarely necessary for special maintenance work.

Documentation
-------------

* Every new feature should be documented.  If it's a new module, don't
  forget to add a new rst file to the API docs.

* Each high-level plotting function should have a small example in
  the ``Examples`` section of the docstring.  This should be as simple as
  possible to demonstrate the method.  More complex examples should go into
  a dedicated example file in the :file:`examples` directory, which will be
  rendered to the examples gallery in the documentation.

* Build the docs and make sure all formatting warnings are addressed.

* See :ref:`documenting-matplotlib` for our documentation style guide.

* If your change is a major new feature, add an entry to
  :file:`doc/users/whats_new.rst`.

* If you change the API in a backward-incompatible way, please
  document it in :file:`doc/api/api_changes.rst`.

PR Review guidelines
====================

* Be patient and `kind <https://youtu.be/tzFWz5fiVKU?t=49m30s>`__ with
  contributors.

* If you have commit rights, then you are trusted to use them.  Please
  help review and merge PRs!

* If you have the rights to set labels, tag the PR with descriptive labels.
  See the `list of labels <https://github.com/matplotlib/matplotlib/labels>`__.

* Set the milestone according to these rules:

  * *New features and API changes* are milestoned for the next point-release
    ``v3.X.0``.

  * *Bugfixes and docstring changes* are milestoned for the next bugfix
    release ``v3.X.Y``

  * *Documentation changes* (all .rst files and examples) are milestoned
    ``v3.X-doc``

  If multiple rules apply, choose the first matching from the above list.

  All of these PRs should target the master branch. The milestone tag triggers
  an :ref:`automatic backport <automated-backports>` for milestones which have
  a corresponding branch.

* Documentation and examples may be merged by the first reviewer.  Use
  the threshold "is this better than it was?" as the review criteria.

* For code changes (anything in ``src`` or ``lib``) at least two
  core developers (those with commit rights) should review all pull
  requests.  If you are the first to review a PR and approve of the
  changes use the GitHub `'approve review'
  <https://help.github.com/articles/reviewing-changes-in-pull-requests/>`__
  tool to mark it as such.  If you are a subsequent reviewer please
  approve the review and if you think no more review is needed, merge
  the PR.

  Ensure that all API changes are documented in
  :file:`doc/api/api_changes` and significant new features have and
  entry in :file:`doc/user/whats_new`.

  - If a PR already has a positive review, a core developer (e.g. the first
    reviewer, but not necessarily) may champion that PR for merging.  In order
    to do so, they should ping all core devs both on GitHub and on the dev
    mailing list, and label the PR with the "Merge with single review?" label.
    Other core devs can then either review the PR and merge or reject it, or
    simply request that it gets a second review before being merged.  If no one
    asks for such a second review within a week, the PR can then be merged on
    the basis of that single review.

    A core dev should only champion one PR at a time and we should try to keep
    the flow of championed PRs reasonable.

* Make sure the Travis, Appveyor, CircleCI, and codecov tests are passing
  before merging.

  - Whenever a pull request is created or updated, Travis and Appveyor
    automatically runs the test suite on all versions of Python
    supported by Matplotlib.  The `tox` support in Matplotlib may be
    useful for testing locally.

* Do not self merge, except for 'small' patches to un-break the CI or
  when another reviewer explicitly allows it (ex, "Approve modulo CI
  passing, may self merge when green").

* Squashing is case-by-case.  The balance is between burden on the
  contributor, keeping a relatively clean history, and keeping a
  history usable for bisecting.  The only time we are really strict
  about it is to eliminate binary files (ex multiple test image
  re-generations) and to remove upstream merges.

* Do not let perfect be the enemy of the good, particularly for
  documentation or example PRs.  If you find yourself making many
  small suggestions, either open a PR against the original branch,
  push changes to the contributor branch, or merge the PR and then
  open a new PR against upstream.

* If you push to a contributor branch leave a comment explaining what
  you did, ex "I took the liberty of pushing a small clean-up PR to
  your branch, thanks for your work.".  If you are going to make
  substantial changes to the code or intent of the PR please check
  with the contributor first.


Branches and Backports
======================


The current active branches are

*master*
  This will be Matplotlib 3.0.  Supports Python 3.5+.

*v2.2.x*
  Maintenance branch for Matplotlib 2.2 LTS.  Supports Python 2.7, 3.4+

*v2.2.N-doc*
  Documentation for the current release.  On a patch release, this will be replaced
  by a properly named branch for the new release.


We always will backport to 2.2.x

- critical bug fixes (segfault, failure to import, things that the
  user can not work around)
- fixes for regressions against 2.0 or 2.1

Everything else (regressions against 1.x versions, bugs/api
inconsistencies the user can work around in their code) are on a
case-by-case basis, should be low-risk, and need someone to advocate
for and shepherd through the backport.

The only changes to be backported to 2.2.N-doc are changes to
``doc``, ``examples``, or ``tutorials``.  Any changes to
``lib`` or ``src`` should not be backported to this branch.

.. _automated-backports:

Automated backports
-------------------

We use meeseeksdev bot to automatically backport merges to the correct
maintenance branch base on the milestone.  To work properly the
milestone must be set before merging.  If you have commit rights, the
bot can also be manually triggered after a merge by leaving a message
``@meeseeksdev backport to BRANCH`` on the PR.  If there are conflicts
meeseekdevs will inform you that the backport needs to be done
manually.

The target branch is configured by putting ``on-merge: backport to
TARGETBRANCH`` in the milestone description on it's own line.

If the bot is not working as expected, please report issues to
`Meeseeksdev <https://github.com/MeeseeksBox/MeeseeksDev>`__.


Manual backports
----------------

When doing backports please copy the form used by meeseekdev,
``Backport PR #XXXX: TITLE OF PR``.  If you need to manually resolve
conflicts make note of them and how you resolved them in the commit
message.

We do a backport from master to v2.2.x assuming:

* ``matplotlib`` is a read-only remote branch of the matplotlib/matplotlib repo

The ``TARGET_SHA`` is the hash of the merge commit you would like to
backport.  This can be read off of the GitHub PR page (in the UI with
the merge notification) or through the git CLI tools.

Assuming that you already have a local branch ``v2.2.x`` (if not, then
``git checkout -b v2.2.x``), and that your remote pointing to
``https://github.com/matplotlib/matplotlib`` is called ``upstream``::

  git fetch upstream
  git checkout v2.2.x  # or include -b if you don't already have this.
  git reset --hard upstream/v2.2.x
  git cherry-pick -m 1 TARGET_SHA
  # resolve conflicts and commit if required

Files with conflicts can be listed by `git status`,
and will have to be fixed by hand (search on ``>>>>>``).  Once
the conflict is resolved, you will have to re-add the file(s) to the branch
and then continue the cherry pick::

  git add lib/matplotlib/conflicted_file.py
  git add lib/matplotlib/conflicted_file2.py
  git cherry-pick --continue

Use your discretion to push directly to upstream or to open a PR; be
sure to push or PR against the `v2.2.x` upstream branch, not `master`!
