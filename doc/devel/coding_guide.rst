.. raw:: html

   <style>
   .checklist { list-style: none; padding: 0; margin: 0; }
   .checklist li { margin-left: 24px; padding-left: 23px;  margin-right: 6px; }
   .checklist li:before { content: "\2610\2001"; margin-left: -24px; }
   .checklist li p {display: inline; }
   </style>

.. _pr-guidelines:

***********************
Pull request guidelines
***********************

Pull requests (PRs) are the mechanism for contributing to Matplotlibs code and
documentation.

Summary for PR authors
======================

.. note::

   * We value contributions from people with all levels of experience. In
     particular if this is your first PR not everything has to be perfect.
     We'll guide you through the PR process.
   * Nevertheless, try to follow the guidelines below as well as you can to
     help make the PR process quick and smooth.
   * Be patient with reviewers. We try our best to respond quickly, but we
     have limited bandwidth. If there is no feedback within a couple of days,
     please ping us by posting a comment to your PR.

When making a PR, pay attention to:

.. rst-class:: checklist

* :ref:`Target the master branch <pr-branch-selection>`.
* Adhere to the :ref:`coding_guidelines`.
* Update the :ref:`documentation <pr-documentation>` if necessary.
* Aim at making the PR as "ready-to-go" as you can. This helps to speed up
  the review process.
* It is ok to open incomplete or work-in-progress PRs if you need help or
  feedback from the developers. You may mark these as
  `draft pull requests <https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_
  on GitHub.
* When updating your PR, instead of adding new commits to fix something, please
  consider amending your initial commit(s) to keep the history clean.
  You can achieve this using

  .. code-block:: bash

     git commit --amend --no-edit
     git push [your-remote-repo] [your-branch] --force-with-lease

See also :ref:`contributing` for how to make a PR.

Summary for PR reviewers
========================

.. note::

   * If you have commit rights, then you are trusted to use them.
     **Please help review and merge PRs!**
   * Be patient and `kind <https://youtu.be/tzFWz5fiVKU?t=49m30s>`__ with
     contributors.

Content topics:

.. rst-class:: checklist

* Is the feature / bugfix reasonable?
* Does the PR conform with the :ref:`coding_guidelines`?
* Is the :ref:`documentation <pr-documentation>` (docstrings, examples,
  what's new, API changes) updated?

Organizational topics:

.. rst-class:: checklist

* Make sure all :ref:`automated tests <pr-automated-tests>` pass.
* The PR should :ref:`target the master branch <pr-branch-selection>`.
* Tag with descriptive :ref:`labels <pr-labels>`.
* Set the :ref:`milestone <pr-milestones>`.
* Keep an eye on the :ref:`number of commits <pr-squashing>`.
* Approve if all of the above topics are handled.
* :ref:`Merge  <pr-merging>` if a sufficient number of approvals is reached.

.. _pr-guidelines-details:

Detailed guidelines
===================

.. _pr-documentation:

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
  document it by adding a file in the relevant subdirectory of
  :file:`doc/api/next_api_changes/`, probably in the ``behavior/``
  subdirectory.

.. _pr-labels:

Labels
------

* If you have the rights to set labels, tag the PR with descriptive labels.
  See the `list of labels <https://github.com/matplotlib/matplotlib/labels>`__.

.. _pr-milestones:

Milestones
----------

* Set the milestone according to these rules:

  * *New features and API changes* are milestoned for the next minor release
    ``v3.X.0``.

  * *Bugfixes and docstring changes* are milestoned for the next patch
    release ``v3.X.Y``

  * *Documentation changes* (all .rst files and examples) are milestoned
    ``v3.X-doc``

  If multiple rules apply, choose the first matching from the above list.

  Setting a milestone does not imply or guarantee that a PR will be merged for that
  release, but if it were to be merged what release it would be in.

  All of these PRs should target the master branch. The milestone tag triggers
  an :ref:`automatic backport <automated-backports>` for milestones which have
  a corresponding branch.

.. _pr-merging:

Merging
-------

* Documentation and examples may be merged by the first reviewer.  Use
  the threshold "is this better than it was?" as the review criteria.

* For code changes (anything in ``src`` or ``lib``) at least two
  core developers (those with commit rights) should review all pull
  requests.  If you are the first to review a PR and approve of the
  changes use the GitHub `'approve review'
  <https://docs.github.com/en/github/collaborating-with-pull-requests/reviewing-changes-in-pull-requests>`__
  tool to mark it as such.  If you are a subsequent reviewer please
  approve the review and if you think no more review is needed, merge
  the PR.

  Ensure that all API changes are documented in a file in one of the
  subdirectories of :file:`doc/api/next_api_changes`, and significant new
  features have an entry in :file:`doc/user/whats_new`.

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

* Do not self merge, except for 'small' patches to un-break the CI or
  when another reviewer explicitly allows it (ex, "Approve modulo CI
  passing, may self merge when green").

.. _pr-automated-tests:

Automated tests
---------------

Whenever a pull request is created or updated, various automated test tools
will run on all supported platforms and versions of Python.

* Make sure the Linting, GitHub Actions, AppVeyor, CircleCI, and Azure
  pipelines are passing before merging (All checks are listed at the bottom of
  the GitHub page of your pull request). Here are some tips for finding the
  cause of the test failure:

  - If *Linting* fails, you have a code style issue, which will be listed
    as annotations on the pull request's diff.
  - If a GitHub Actions or AppVeyor run fails, search the log for ``FAILURES``.
    The subsequent section will contain information on the failed tests.
  - If CircleCI fails, likely you have some reStructuredText style issue in
    the docs. Search the CircleCI log for ``WARNING``.
  - If Azure pipelines fail with an image comparison error, you can find the
    images as *artifacts* of the Azure job:

    - Click *Details* on the check on the GitHub PR page.
    - Click *View more details on Azure Pipelines* to go to Azure.
    - On the overview page *artifacts* are listed in the section *Related*.


* Codecov and LGTM are currently for information only. Their failure is not
  necessarily a blocker.

* tox_ is not used in the automated testing. It is supported for testing
  locally.

  .. _tox: https://tox.readthedocs.io/

* If you know your changes do not need to be tested (this is very rare!), all
  CIs can be skipped for a given commit by including ``[ci skip]`` or
  ``[skip ci]`` in the commit message. If you know only a subset of CIs need
  to be run (e.g., if you are changing some block of plain reStructuredText and
  want only CircleCI to run to render the result), individual CIs can be
  skipped on individual commits as well by using the following substrings
  in commit messages:

  - GitHub Actions: ``[skip actions]``
  - AppVeyor: ``[skip appveyor]`` (must be in the first line of the commit)
  - Azure Pipelines: ``[skip azp]``
  - CircleCI: ``[skip circle]``

.. _pr-squashing:

Number of commits and squashing
-------------------------------

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


.. _branches_and_backports:

Branches and backports
======================

Current branches
----------------
The current active branches are

*master*
  The current development version. Future minor releases (*v3.N.0*) will be
  branched from this. Supports Python 3.7+.

*v3.N.x*
  Maintenance branch for Matplotlib 3.N. Future patch releases will be
  branched from this.  Supports Python 3.6+.

*v3.N.M-doc*
  Documentation for the current release.  On a patch release, this will be
  replaced by a properly named branch for the new release.


.. _pr-branch-selection:

Branch selection for pull requests
----------------------------------

Generally, all pull requests should target the master branch.

Other branches are fed through :ref:`automatic <automated-backports>` or
:ref:`manual <manual-backports>`. Directly
targeting other branches is only rarely necessary for special maintenance
work.

.. backport_strategy:

Backport strategy
-----------------

We will always backport to the patch release branch (*v3.N.x*):

- critical bug fixes (segfault, failure to import, things that the
  user can not work around)
- fixes for regressions against the last two releases.

Everything else (regressions against older releases, bugs/api
inconsistencies the user can work around in their code) are on a
case-by-case basis, should be low-risk, and need someone to advocate
for and shepherd through the backport.

The only changes to be backported to the documentation branch (*v3.N.M-doc*)
are changes to :file:`doc`, :file:`examples`, or :file:`tutorials`.
Any changes to :file:`lib` or :file:`src` including docstring-only changes
should not be backported to this branch.


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


.. _manual-backports:

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
``https://github.com/matplotlib/matplotlib`` is called ``upstream``:

.. code-block:: bash

   git fetch upstream
   git checkout v2.2.x  # or include -b if you don't already have this.
   git reset --hard upstream/v2.2.x
   git cherry-pick -m 1 TARGET_SHA
   # resolve conflicts and commit if required

Files with conflicts can be listed by ``git status``,
and will have to be fixed by hand (search on ``>>>>>``).  Once
the conflict is resolved, you will have to re-add the file(s) to the branch
and then continue the cherry pick:

.. code-block:: bash

   git add lib/matplotlib/conflicted_file.py
   git add lib/matplotlib/conflicted_file2.py
   git cherry-pick --continue

Use your discretion to push directly to upstream or to open a PR; be
sure to push or PR against the ``v2.2.x`` upstream branch, not ``master``!
