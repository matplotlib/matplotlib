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

`Pull requests (PRs) on GitHub
<https://docs.github.com/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`__
are the mechanism for contributing to Matplotlib's code and documentation.

It is recommended to check that your contribution complies with the following
rules before submitting a pull request:

* If your pull request addresses an issue, please use the title to describe the
  issue (e.g. "Add ability to plot timedeltas") and mention the issue number
  in the pull request description to ensure that a link is created to the
  original issue (e.g. "Closes #8869" or "Fixes #8869"). This will ensure the
  original issue mentioned is automatically closed when your PR is merged. See
  `the GitHub documentation
  <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`__
  for more details.

* Formatting should follow the recommendations of PEP8_, as enforced by
  flake8_. Matplotlib modifies PEP8 to extend the maximum line length to 88
  characters. You can check flake8 compliance from the command line with ::

    python -m pip install flake8
    flake8 /path/to/module.py

  or your editor may provide integration with it.  Note that Matplotlib
  intentionally does not use the black_ auto-formatter (1__), in particular due
  to its inability to understand the semantics of mathematical expressions
  (2__, 3__).

  .. _PEP8: https://www.python.org/dev/peps/pep-0008/
  .. _flake8: https://flake8.pycqa.org/
  .. _black: https://black.readthedocs.io/
  .. __: https://github.com/matplotlib/matplotlib/issues/18796
  .. __: https://github.com/psf/black/issues/148
  .. __: https://github.com/psf/black/issues/1984

* All public methods should have informative docstrings with sample usage when
  appropriate. Use the :ref:`docstring standards <writing-docstrings>`.

* For high-level plotting functions, consider adding a simple example either in
  the ``Example`` section of the docstring or the
  :ref:`examples gallery <gallery>`.

* Changes (both new features and bugfixes) should have good test coverage. See
  :ref:`testing` for more details.

* Import the following modules using the standard scipy conventions::

     import numpy as np
     import numpy.ma as ma
     import matplotlib as mpl
     import matplotlib.pyplot as plt
     import matplotlib.cbook as cbook
     import matplotlib.patches as mpatches

  In general, Matplotlib modules should **not** import `.rcParams` using ``from
  matplotlib import rcParams``, but rather access it as ``mpl.rcParams``.  This
  is because some modules are imported very early, before the `.rcParams`
  singleton is constructed.

* If your change is a major new feature, add an entry to the ``What's new``
  section by adding a new file in ``doc/users/next_whats_new`` (see
  :file:`doc/users/next_whats_new/README.rst` for more information).

* If you change the API in a backward-incompatible way, please document it in
  :file:`doc/api/next_api_changes/behavior`, by adding a new file with the
  naming convention ``99999-ABC.rst`` where the pull request number is followed
  by the contributor's initials. (see :file:`doc/api/api_changes.rst` for more
  information)

* If you add new public API or change public API, update or add the
  corresponding type hints. Most often this is found in the corresponding
  ``.pyi`` file for the ``.py`` file which was edited. Changes in ``pyplot.py``
  are type hinted inline.

* See below for additional points about :ref:`keyword-argument-processing`, if
  applicable for your pull request.

.. note::

    The current state of the Matplotlib code base is not compliant with all
    of these guidelines, but we expect that enforcing these constraints on all
    new contributions will move the overall code base quality in the right
    direction.


.. seealso::

  * :ref:`coding_guidelines`
  * :ref:`testing`
  * :ref:`documenting-matplotlib`



Summary for pull request authors
================================

.. note::

   * We value contributions from people with all levels of experience. In
     particular if this is your first PR not everything has to be perfect.
     We'll guide you through the PR process.
   * Nevertheless, please try to follow the guidelines below as well as you can to
     help make the PR process quick and smooth.
   * Be patient with reviewers. We try our best to respond quickly, but we
     have limited bandwidth. If there is no feedback within a couple of days,
     please ping us by posting a comment to your PR.

When making a PR, pay attention to:

.. rst-class:: checklist

* :ref:`Target the main branch <pr-branch-selection>`.
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
  You can achieve this by using

  .. code-block:: bash

     git commit --amend --no-edit
     git push [your-remote-repo] [your-branch] --force-with-lease

See also :ref:`contributing` for how to make a PR.

Summary for pull request reviewers
==================================

.. redirect-from:: /devel/maintainer_workflow

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
* Is the change purely stylistic? Generally, such changes are discouraged when
  not part of other non-stylistic work because it obscures the git history of
  functional changes to the code. Reflowing a method or docstring as part of a
  larger refactor/rewrite is acceptable.


Organizational topics:

.. rst-class:: checklist

* Make sure all :ref:`automated tests <pr-automated-tests>` pass.
* The PR should :ref:`target the main branch <pr-branch-selection>`.
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

.. _release_notes:

New features and API changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When adding a major new feature or changing the API in a backward incompatible
way, please document it by including a versioning directive in the docstring
and adding an entry to the folder for either the what's new or API change notes.

+-------------------+-----------------------------+----------------------------------+
| for this addition | include this directive      | create entry in this folder      |
+===================+=============================+==================================+
| new feature       | ``.. versionadded:: 3.N``   | :file:`doc/users/next_whats_new/`|
+-------------------+-----------------------------+----------------------------------+
| API change        | ``.. versionchanged:: 3.N`` | :file:`doc/api/next_api_changes/`|
|                   |                             |                                  |
|                   |                             | probably in ``behavior/``        |
+-------------------+-----------------------------+----------------------------------+

The directives should be placed at the end of a description block. For example::

  class Foo:
      """
      This is the summary.

      Followed by a longer description block.

      Consisting of multiple lines and paragraphs.

      .. versionadded:: 3.5

      Parameters
      ----------
      a : int
          The first parameter.
      b: bool, default: False
          This was added later.

          .. versionadded:: 3.6
      """

      def set_b(b):
          """
          Set b.

          .. versionadded:: 3.6

          Parameters
          ----------
          b: bool

For classes and functions, the directive should be placed before the
*Parameters* section. For parameters, the directive should be placed at the
end of the parameter description. The patch release version is omitted and
the directive should not be added to entire modules.

.. _pr-labels:

Labels
------

* If you have the rights to set labels, tag the PR with descriptive labels.
  See the `list of labels <https://github.com/matplotlib/matplotlib/labels>`__.
* If the PR makes changes to the wheel building Action, add the
  "Run cibuildwheel" label to enable testing wheels.

.. _pr-milestones:

Milestones
----------

Set the milestone according to these guidelines:

* *New features and API changes* are milestoned for the next minor release
  ``v3.N.0``.

* *Bugfixes, tests for released code, and docstring changes* may be milestoned
  for the next patch release ``v3.N.M``.

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
Before being merged, a PR should pass the :ref:`automated-tests`. If you are unsure why a test is failing, ask on the PR or in our `chat space <https://gitter.im/matplotlib/matplotlib>`_

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

*main*
  The current development version. Future minor releases (*v3.N.0*) will be
  branched from this.

*v3.N.x*
  Maintenance branch for Matplotlib 3.N. Future patch releases will be
  branched from this.

*v3.N.M-doc*
  Documentation for the current release.  On a patch release, this will be
  replaced by a properly named branch for the new release.


.. _pr-branch-selection:

Branch selection for pull requests
----------------------------------

Generally, all pull requests should target the main branch.

Other branches are fed through :ref:`automatic <automated-backports>` or
:ref:`manual <manual-backports>`. Directly
targeting other branches is only rarely necessary for special maintenance
work.

.. _backport-strategy:

Backport strategy
-----------------

Backports to the patch release branch (*v3.N.x*) are the changes that will be
included in the next patch (aka bug-fix) release.  The goal of the patch
releases is to fix bugs without adding any new regressions or behavior changes.
We will always attempt to backport:

- critical bug fixes (segfault, failure to import, things that the
  user cannot work around)
- fixes for regressions introduced in the last two minor releases

and may attempt to backport fixes for regressions introduced in older releases.

In the case where the backport is not clean, for example if the bug fix is
built on top of other code changes we do not want to backport, balance the
effort and risk of re-implementing the bug fix vs the severity of the bug.
When in doubt, err on the side of not backporting.

When backporting a Pull Request fails or is declined, re-milestone the original
PR to the next minor release and leave a comment explaining why.

The only changes backported to the documentation branch (*v3.N.M-doc*)
are changes to :file:`doc` or :file:`galleries`.  Any changes to :file:`lib`
or :file:`src`, including docstring-only changes, must not be backported to
this branch.


.. _automated-backports:

Automated backports
-------------------

We use MeeseeksDev bot to automatically backport merges to the correct
maintenance branch base on the milestone.  To work properly the
milestone must be set before merging.  If you have commit rights, the
bot can also be manually triggered after a merge by leaving a message
``@meeseeksdev backport to BRANCH`` on the PR.  If there are conflicts
MeeseeksDev will inform you that the backport needs to be done
manually.

The target branch is configured by putting ``on-merge: backport to
TARGETBRANCH`` in the milestone description on it's own line.

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
