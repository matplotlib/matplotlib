.. _pr-guide-maintainers:

********************************
Pull request guide for reviewers
********************************

Summary for pull request reviewers
==================================

.. redirect-from:: /devel/maintainer_workflow

**Please help review and merge PRs!**

If you have commit rights, then you are trusted to use them. Please be patient
and `kind <https://youtu.be/tzFWz5fiVKU?t=49m30s>`__ with contributors.

When reviewing, please ensure that the pull request satisfies the following
requirements before merging it:

Content
-------

.. rst-class:: checklist

* Is the feature / bugfix reasonable?
* Does the PR conform with the :ref:`coding_guidelines`?
* Is the :ref:`documentation <pr-documentation>` (docstrings, examples,
  what's new, API changes) updated?
* Is the change purely stylistic? Generally, such changes are discouraged when
  not part of other non-stylistic work because it obscures the git history of
  functional changes to the code. Reflowing a method or docstring as part of a
  larger refactor/rewrite is acceptable.

Workflow
--------
.. rst-class:: checklist

* Make sure all :ref:`automated tests <automated-tests>` pass.
* The PR should target the ``main`` branch.
* Tag with descriptive :ref:`labels <pr-labels>`.
* Set the :ref:`milestone <pr-milestones>`.
* :ref:`Review <pr-review>` the contents.
* Approve if all of the above topics are handled.
* Keep an eye on the :ref:`number of commits <pr-squashing>`.
* :ref:`Merge <pr-merging-maintainers>` if a
  :ref:`sufficient number of approvals <pr-approval>` is reached.

Draft PRs
---------

Authors may create a `draft PR`_ (or change to draft status later) if the code
is not yet ready for a regular full review. Typical use cases are posting code
as a basis for discussion or signalling that you intend to rework the code as
a result of feedback. Authors should clearly communicate why the PR has draft
status and what needs to be done to make it ready for review. In particular,
they should explicitly ask for targeted feedback if needed. By default,
reviewers will not look at the code of a draft PR and only respond to specific
questions by the author.

.. _draft PR: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests

.. _pr-labels:

Labels
------

* If you have the rights to set labels, tag the PR with descriptive labels.
  See the `list of labels <https://github.com/matplotlib/matplotlib/labels>`__.
* If the PR makes changes to the wheel building Action, add the
  "Run cibuildwheel" label to enable testing wheels.
* If the PR does not yet have the quality and clarity needed for an effective
  review, you can use the ``status: autoclose candidate`` label. This will
  trigger a two-weeks countdown after which the PR will be automatically closed
  if no further improvements have been made. See
  `the autoclose workflow <https://github.com/matplotlib/matplotlib/blob/main/.github/workflows/autoclose_comment.yml>`__
  for more details.

.. _pr-milestones:

Milestones
----------

Set the milestone according to these guidelines:

* *New features and API changes* are milestoned for the next meso release
  ``v3.N.0``.
* *Bugfixes, tests for released code, and docstring changes* may be milestoned
  for the next micro release ``v3.N.M``.
* *Documentation changes* (only .rst files and examples) may be milestoned
  ``v3.N-doc``.

If multiple rules apply, choose the first matching from the above list.  See
:ref:`backport-strategy` for detailed guidance on what should or should not be
backported.

The milestone marks the release a PR should go into.  It states intent, but can
be changed because of release planning or re-evaluation of the PR scope and
maturity.

All Pull Requests should target the main branch. The milestone tag triggers
an :ref:`automatic backport <automated-backports>` for milestones which have
a corresponding branch.

.. _pr-review:

Review
------

* Do not let perfect be the enemy of the good, particularly for
  documentation or example PRs.  If you find yourself making many
  small suggestions, either open a PR against the original branch,
  push changes to the contributor branch, or merge the PR and then
  open a new PR against upstream.

* If you push to a contributor branch, leave a comment explaining what
  you did, ex "I took the liberty of pushing a small clean-up PR to
  your branch, thanks for your work.".  If you are going to make
  substantial changes to the code or intent of the PR please check
  with the contributor first.

* If you find yourself spending too much time on a PR, or feeling frustrated,
  it's ok to step back. You can ask for help from other reviewers, or if you are
  the only reviewer, you can ask the contributor to find another reviewer or to
  wait until you have more time. Make sure to communicate with the contributor
  to set the right expectations, e.g. "I currently don't have the bandwidth to
  review this PR, but will try to loop someone else in." If you feel like this
  PR is not a good fit for the project, you can close it with an explanation or
  add the "status: autoclose candidate" label to trigger the autoclose workflow.

.. _pr-approval:

Approval
--------
As a guiding principle, we require two `approvals`_ from core developers (those
with commit rights) before merging a pull request. This two-pairs-of-eyes
strategy shall ensure a consistent project direction and prevent accidental
mistakes. It is permissible to merge with one approval if the change is not
fundamental and can easily be reverted at any time in the future.

.. _approvals: https://docs.github.com/en/github/collaborating-with-pull-requests/reviewing-changes-in-pull-requests

Some explicit rules following from this:

* Small and medium sized *Documentation and examples* may be merged with a single approval.
  Use the threshold "is this better than it was?" as the review criteria. Large documentation
  PRs (e.g. adds large new sections or new rst pages) require two reviews.

* Minor *infrastructure updates*, e.g. temporary pinning of broken dependencies
  or small changes to the CI configuration, may be merged with a single
  approval.

* *Code changes* (anything in ``src`` or ``lib``) must have two approvals.

  Ensure that all API changes are documented in a file in one of the
  subdirectories of :file:`doc/api/next_api_changes`, and significant new
  features have an entry in :file:`doc/release/next_whats_new`.

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

.. _pr-merging:

Merging
-------
After giving the last required :ref:`approval <pr-approval>`, the author of the
approval should merge the PR. PR authors should not self-merge except for when
another reviewer explicitly allows it (e.g., "Approve modulo CI passing, may
self-merge when green", or "Take or leave the comments. You may self merge").

.. _branches_and_backports:

Branches and backports
======================

Current branches
----------------
The current active branches are

*main*
  The current development version. Future meso (*v3.N.0*) or macro (*v4.0.0*) will be
  branched from this.

*v3.N.x*
  Maintenance branch for Matplotlib 3.N. Future micro releases will be
  tagged from this.

*v3.N.M-doc*
  Documentation for the current micro release.  On a micro release, this will be
  replaced by a properly named branch for the new release.

.. _backport-strategy:

Backport strategy
-----------------

Backports to the micro release branch (*v3.N.x*) are the changes that will be
included in the next patch (aka bug-fix) release.  The goal of the patch
releases is to fix bugs without adding any new regressions or behavior changes.
We will always attempt to backport:

- critical bug fixes (segfault, failure to import, things that the
  user cannot work around)
- fixes for regressions introduced in the last two meso releases

and may attempt to backport fixes for regressions introduced in older releases.

In the case where the backport is not clean, for example if the bug fix is
built on top of other code changes we do not want to backport, balance the
effort and risk of re-implementing the bug fix vs the severity of the bug.
When in doubt, err on the side of not backporting.

When backporting a Pull Request fails or is declined, re-milestone the original
PR to the next meso release and leave a comment explaining why.

The only changes backported to the documentation branch (*v3.N.M-doc*)
are changes to :file:`doc` or :file:`galleries`.  Any changes to :file:`lib`
or :file:`src`, including docstring-only changes, must not be backported to
this branch.


.. _automated-backports:

Automated backports
-------------------

We use MeeseeksDev bot to automatically backport merges to the correct
maintenance branch based on the milestone.  To work properly the
milestone must be set before merging.  If you have commit rights, the
bot can also be manually triggered after a merge by leaving a message
``@meeseeksdev backport to BRANCH`` on the PR.  If there are conflicts
MeeseeksDev will inform you that the backport needs to be done
manually.

The target branch is configured by putting ``on-merge: backport to
TARGETBRANCH`` in the milestone description on its own line.

If the bot is not working as expected, please report issues to
`MeeseeksDev <https://github.com/MeeseeksBox/MeeseeksDev>`__.


.. _manual-backports:

Manual backports
----------------

When doing backports please copy the form used by MeeseeksDev,
``Backport PR #XXXX: TITLE OF PR``.  If you need to manually resolve
conflicts make note of them and how you resolved them in the commit
message.

We do a backport from main to v2.2.x assuming:

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
sure to push or PR against the ``v2.2.x`` upstream branch, not ``main``!
