.. _pr-guidelines:

****************************
Pull request authoring guide
****************************

`Pull requests (PRs) on GitHub
<https://docs.github.com/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`__
are the mechanism for contributing to Matplotlib's code and documentation.

We value contributions from people with all levels of experience. In particular,
if this is your first PR not everything has to be perfect. We'll guide you
through the PR process. Nevertheless, please try to follow our guidelines as
well as you can to help make the PR process quick and smooth.

.. admonition:: Note

  If your pull request is incomplete or a work-in-progress, please mark it as a
  `draft pull request <draft PR>`__ on GitHub and specify what feedback from
  the developers would be helpful.

.. _draft PR: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests

Please be patient with reviewers. We try our best to respond quickly, but we
have limited bandwidth. If there is no feedback within a couple of days, please
ping us by posting a comment to your PR or reaching out on one of our
:ref:`communication channels <communication-channels>`.

.. _pr-author-guidelines:

Summary for pull request authors
================================

We recommend that you check that your contribution complies with the following
guidelines before submitting a pull request:

.. rst-class:: checklist

* Changes, both new features and bugfixes, should have good test coverage. See
  :ref:`testing` for more details.
* Update the :ref:`documentation <pr-documentation>` if necessary.
* All public methods should have informative docstrings with sample usage when
  appropriate. Use the :ref:`docstring standards <writing-docstrings>`.
* For high-level plotting functions, consider
  :ref:`adding a small example <writing-examples-and-tutorials>` to the
  :ref:`examples gallery <gallery>`.
* If you add a new feature or change the API in a backward-incompatible
  way, please document it as described in :ref:`api_changes`.
* Code should follow our conventions as documented in our
  :ref:`coding_guidelines`.
* When adding or changing public function signatures, add
  :ref:`type hints <type-hints>`.
* When adding keyword arguments, see our guide to
  :ref:`keyword-argument-processing`.

When opening a pull request on Github, please ensure that:

.. rst-class:: checklist

* Changes were made on a :ref:`feature branch <git-overview>` (i.e. not on the
  ``main`` branch.)
* :ref:`prek <pre-commit-hooks>` checks for spelling, formatting, etc pass
* The pull request targets the :ref:`main branch <pr-branch-selection>`
* If your pull request addresses an issue, please use the title to describe the
  issue (e.g. "Add ability to plot timedeltas") and mention the issue number
  in the pull request description to ensure that a link is created to the
  original issue (e.g. "Closes #8869" or "Fixes #8869"). This will ensure the
  original issue mentioned is automatically closed when your PR is merged. For more
  details, see
  `linking an issue and pull request <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`__.
* :ref:`pr-automated-tests` pass
* You follow the Pull Request template, which includes a checklist of items to
  verify before submitting your PR and an :ref:`AI Disclosure <generative_ai>`
  section.

For guidance on creating and managing a pull request, please see our
:ref:`contributing <contributing>` and :ref:`pull request workflow <edit-flow>`
guides.

.. _pr-branch-selection:

Branch selection for pull requests
==================================

Generally, all pull requests should target the main branch.

Other branches are fed through :ref:`automatic <automated-backports>` or
:ref:`manual <manual-backports>`. Directly targeting other branches is only
rarely necessary for special maintenance work.

.. _pr-documentation:

Documentation
=============

* Every new feature should be documented. If it's a new module, don't
  forget to add a new rst file to the API docs.

* Each high-level plotting function should have a small example in
  the ``Examples`` section of the docstring.  This should be as simple as
  possible to demonstrate the method.  More complex examples should go into
  a dedicated example file in the :file:`examples` directory, which will be
  rendered to the examples gallery in the documentation.

* :ref:`Build the docs <documenting-matplotlib>` and make sure all formatting
  warnings are addressed.

* See how to :ref:`write documentation <write-documentation>` for Matplotlib and
  consult our :ref:`documentation style guide <document_style>`.

Next steps
==========

After you submit your PR, one or more reviewers will address it. This can
include providing feedback, requesting changes, adding labels, or approving and
merging the PR.

If the PR does not yet have the quality and clarity needed for an effective
review, or if you fail to fill out the PR template correctly (including not
providing an AI Disclosure), your PR may be labeled as
``status: autoclose candidate``. This will trigger a two-weeks countdown after
which the PR will be automatically closed if no further improvements have been
made. See `the autoclose workflow <https://github.com/matplotlib/matplotlib/blob/main/.github/workflows/autoclose_comment.yml>`__
for more details.

.. _pr-automated-tests:

Automated tests
---------------
Before being merged, a PR should pass the :ref:`automated-tests`. If you are
unsure why a test is failing, ask on the PR or in one of our
:ref:`communication-channels`.

Approval
--------
As a guiding principle, we require two `approvals`_ from core developers (those
with commit rights) before merging a pull request. This two-pairs-of-eyes
strategy shall ensure a consistent project direction and prevent accidental
mistakes. It is permissible to merge with one approval if the change is not
fundamental and can easily be reverted at any time in the future.

.. _approvals: https://docs.github.com/en/github/collaborating-with-pull-requests/reviewing-changes-in-pull-requests

See the :ref:`pull request guide for reviewers <pr-guide-maintainers>` for more
details on this process.

.. _pr-merging:

Merging
-------
After giving the last required approval, the author of the
approval should merge the PR. PR authors should not self-merge except for when
another reviewer explicitly allows it (e.g., "Approve modulo CI passing, may
self-merge when green", or "Take or leave the comments. You may self merge").

.. _pr-squashing:

Number of commits and squashing
-------------------------------

When merging your Pull Request, your commits may be squashed (i.e., combined
into a single commit) to keep the repository history clean. Squashing is decided
on a case-by-case basis. The balance is between burden on the contributor,
keeping a relatively clean history, and keeping a history usable for bisecting.
The only time we are really strict about it is to eliminate binary files (e.g.,
multiple test image re-generations) and to remove upstream merges.
