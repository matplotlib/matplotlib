.. _reviewers-guide:

********************
Reviewers guideline
********************

.. _pull-request-checklist:

Pull request checklist
======================

Branch selection
----------------

* In general, simple bugfixes that are unlikely to introduce new bugs
  of their own should be merged onto the maintenance branch.  New
  features, or anything that changes the API, should be made against
  master.  The rules are fuzzy here -- when in doubt, target master.

* Once changes are merged into the maintenance branch, they should
  be merged into master.

Documentation
-------------

* Every new feature should be documented.  If it's a new module, don't
  forget to add a new rst file to the API docs.

* Each high-level plotting function should have a small example in
  the `Example` section of the docstring.  This should be as simple as
  possible to demonstrate the method.  More complex examples should go
  in the `examples` section of the documentation.

* Build the docs and make sure all formatting warnings are addressed.

* See :ref:`documenting-matplotlib` for our documentation style guide.

* If your change is a major new feature, add an entry to
  :file:`doc/users/whats_new.rst`.

* If you change the API in a backward-incompatible way, please
  document it in :file:`doc/api/api_changes.rst`.

PR Review guidelines
====================

* If you have commit rights, then you are trusted to use them.  Please
  help review and merge PRs!

* Two developers (those with commit rights) should review all pull
  requests.  If you are the first to review a PR and approve of
  the changes, please edit the title to include ``'[MRG+1]'`` and use
  the github `'approve review'
  <https://help.github.com/articles/reviewing-changes-in-pull-requests/>`__
  tool to mark it as such.  If you are a subsequent reviewer and you
  approve, either merge (and backport if needed) or select ``'approve review'`` and 
  increment the number in the title to ask for further review.  
  If you do the merge, please remove the ``'[MRG+N']`` prefix.

* Make sure the Travis tests are passing before merging.

  - Whenever a pull request is created or updated, Travis automatically runs 
    the test suite on all versions of Python supported by Matplotlib.
    The `tox` support in Matplotlib may be useful for testing locally.

* Do not self merge, except for 'small' patches to un-break the CI.

* Squashing is case-by-case.  The balance is between burden on the
  contributor, keeping a relatively clean history, and keeping a
  history usable for bisecting.  The only time we are really strict
  about it is to eliminate binary files (ex multiple test image
  re-generations) and to remove upstream merges.

* Be patient with contributors.

* Do not let perfect be the enemy of the good, particularly for
  documentation or example PRs.  If you find yourself making many
  small suggestions, either open a PR against the original branch or
  merge the PR and then open a new PR against upstream.


Backports
=========


When doing backports please include the branch you backported the
commit to along with the SHA in a comment on the original PR.

We do a backport from master to v2.0.x assuming:

* ``matplotlib`` is a read-only remote branch of the matplotlib/matplotlib repo 

* ``DANGER`` is a read/write remote branch of the matplotlib/matplotlib repo

The ``TARGET_SHA`` is the hash of the merge commit you would like to
backport.  This can be read off of the github PR page (in the UI with
the merge notification) or through the git CLI tools.::

  git fetch matplotlib
  git checkout v2.0.x
  git merge --ff-only matplotlib/v2.0.x
  git cherry-pick -m 1 TARGET_SHA
  git log --graph --decorate  # to look at it
  # local tests? (use your judgment)
  git push DANGER v2.0.x
  # leave a comment on PR noting sha of the resulting commit
  # from the cherry-pick + branch it was moved to

These commands work on git 2.7.1.
