.. highlight:: bash

.. _release-guide:

=============
Release guide
=============


.. admonition::  This document is only relevant for Matplotlib release managers.

   A guide for developers who are doing a Matplotlib release.


.. note::

   This assumes that a read-only remote for the canonical repository is
   ``remote`` and a read/write remote is ``DANGER``


.. _release_feature_freeze:

Making the release branch
=========================

When a new minor release (vX.Y.0) is approaching, a new release branch must be made.
When precisely this should happen is up to the release manager, but this point is where
most new features intended for the minor release are merged and you are entering a
feature freeze (i.e. newly implemented features will be going into vX.Y+1).
This does not necessarily mean that no further changes will be made prior to release,
just that those changes will be made using the backport system.

For an upcoming ``v3.7.0`` release, first create the branch::

   git switch main
   git pull
   git switch -c v3.7.x
   git push DANGER v3.7.x

Update the ``v3.7.0`` milestone so that the description reads::

   New features and API changes

   on-merge: backport to v3.7.x

Micro versions should instead read::

   Bugfixes and docstring changes

   on-merge: backport to v3.7.x

Check all active milestones for consistency. Older milestones should also backport
to higher minor versions (e.g. ``v3.6.3`` and ``v3.6-doc`` should backport to both
``v3.6.x`` and ``v3.7.x`` once the ``v3.7.x`` branch exists and while PR backports are
still targeting ``v3.6.x``)

Create the milestone for the next-next minor release (i.e. ``v3.9.0``, as ``v3.8.0``
should already exist). While most active items should go in the next minor release,
this milestone can help with longer term planning, especially around deprecation
cycles.

.. _release-testing:

Testing
=======

We use `GitHub Actions <https://github.com/matplotlib/matplotlib/actions>`__
for continuous integration.  When preparing for a release, the final tagged
commit should be tested locally before it is uploaded::

   pytest -n 8 .


In addition the following test should be run and manually inspected::

   python tools/memleak.py agg 1000 agg.pdf

Run the User Acceptance Tests for the NBAgg and ipympl backends::

   jupyter notebook lib/matplotlib/backends/web_backend/nbagg_uat.ipynb

For ipympl, restart the kernel, add a cell for ``%matplotlib widget`` and do
not run the cell with ``matplotlib.use('nbagg')``. Tests which check
``connection_info``, use ``reshow``, or test the OO interface are not expected
to work for ``ipympl``.

.. _release_ghstats:

GitHub statistics
=================

We automatically extract GitHub issue, PRs, and authors from GitHub via the API. To
prepare this list:

1. Archive the existing GitHub statistics page.

   a. Copy the current :file:`doc/users/github_stats.rst` to
      :file:`doc/users/prev_whats_new/github_stats_{X}.{Y}.{Z}.rst`.
   b. Change the link target at the top of the file.
   c. Remove the "Previous GitHub Stats" section at the end.

   For example, when updating from v3.7.0 to v3.7.1::

      cp doc/users/github_stats.rst doc/users/prev_whats_new/github_stats_3.7.0.rst
      $EDITOR doc/users/prev_whats_new/github_stats_3.7.0.rst
      # Change contents as noted above.
      git add doc/users/prev_whats_new/github_stats_3.7.0.rst

2. Re-generate the updated stats::

       python tools/github_stats.py --since-tag v3.7.0 --milestone=v3.7.1 \
           --project 'matplotlib/matplotlib' --links > doc/users/github_stats.rst

3. Review and commit changes. Some issue/PR titles may not be valid reST (the most
   common issue is ``*`` which is interpreted as unclosed markup).

.. note::

   Make sure you authenticate against the GitHub API. If you do not, you will get
   blocked by GitHub for going over the API rate limits. You can authenticate in one of
   two ways:

   * using the ``keyring`` package; ``pip install keyring`` and then when
     running the stats script, you will be prompted for user name and password,
     that will be stored in your system keyring, or,
   * using a personal access token; generate a new token `on this GitHub page
     <https://github.com/settings/tokens>`__ with the ``repo:public_repo``
     scope and place the token in :file:`~/.ghoauth`.


.. _release_chkdocs:

Update and validate the docs
============================

Merge ``*-doc`` branch
----------------------

Merge the most recent 'doc' branch (e.g., ``v3.7.0-doc``) into the branch you
are going to tag on and delete the doc branch on GitHub.

Update supported versions in Security Policy
--------------------------------------------

When making major or minor releases, update the supported versions in the Security
Policy in :file:`SECURITY.md`.

For minor version release update the table in :file:`SECURITY.md` to specify that the
two most recent minor releases in the current major version series are supported.

For a major version release update the table in :file:`SECURITY.md` to specify that the
last minor version in the previous major version series is still supported. Dropping
support for the last version of a major version series will be handled on an ad-hoc
basis.

Update release notes
--------------------

What's new
^^^^^^^^^^

*Only needed for major and minor releases. Bugfix releases should not have new
features.*

Merge the contents of all the files in :file:`doc/users/next_whats_new/` into a single
file :file:`doc/users/prev_whats_new/whats_new_{X}.{Y}.0.rst` and delete the individual
files.

API changes
^^^^^^^^^^^

*Primarily needed for major and minor releases. We may sometimes have API
changes in bugfix releases.*

Merge the contents of all the files in :file:`doc/api/next_api_changes/` into a single
file :file:`doc/api/prev_api_changes/api_changes_{X}.{Y}.{Z}.rst` and delete the
individual files.

Release notes TOC
^^^^^^^^^^^^^^^^^

Update :file:`doc/users/release_notes.rst`:

- For major and minor releases add a new section

  .. code:: rst

     X.Y
     ===
     .. toctree::
         :maxdepth: 1

         prev_whats_new/whats_new_X.Y.0.rst
         ../api/prev_api_changes/api_changes_X.Y.0.rst
         prev_whats_new/github_stats_X.Y.0.rst
- For bugfix releases add the GitHub stats and (if present) the API changes to
  the existing X.Y section

  .. code:: rst

     ../api/prev_api_changes/api_changes_X.Y.Z.rst
     prev_whats_new/github_stats_X.Y.Z.rst

Update version switcher
^^^^^^^^^^^^^^^^^^^^^^^

Update ``doc/_static/switcher.json``:

- If a bugfix release, :samp:`{X}.{Y}.{Z}`, no changes are needed.
- If a major release, :samp:`{X}.{Y}.0`, change the name of :samp:`name: {X}.{Y+1}
  (dev)` and :samp:`name: {X}.{Y} (stable)` as well as adding a new version for the
  previous stable (:samp:`name: {X}.{Y-1}`).

Verify that docs build
----------------------

Finally, make sure that the docs build cleanly::

  make -Cdoc O=-j$(nproc) html latexpdf

After the docs are built, check that all of the links, internal and external, are still
valid. We use ``linkchecker`` for this::

  pip install linkchecker
  pushd doc/build/html
  linkchecker index.html --check-extern
  popd

Address any issues which may arise. The internal links are checked on Circle CI, so this
should only flag failed external links.


.. _release_tag:

Create release commit and tag
=============================

To create the tag, first create an empty commit with a very terse set of the release
notes in the commit message::

  git commit --allow-empty

and then create a signed, annotated tag with the same text in the body message::

  git tag -a -s v3.7.0

which will prompt you for your GPG key password and an annotation.  For pre-releases it
is important to follow :pep:`440` so that the build artifacts will sort correctly in
PyPI.

To prevent issues with any down-stream builders which download the tarball from GitHub
it is important to move all branches away from the commit with the tag [#]_::

  git commit --allow-empty

Finally, push the tag to GitHub::

  git push DANGER v3.7.x v3.7.0

Congratulations, the scariest part is done!
This assumes the release branch has already been made.
Usually this is done at the time of feature freeze for a minor release (which often
coincides with the last patch release of the previous minor version)

.. [#] The tarball that is provided by GitHub is produced using `git archive`_.
       We use setuptools_scm_ which uses a format string in
       :file:`lib/matplotlib/_version.py` to have ``git`` insert a
       list of references to exported commit (see
       :file:`.gitattributes` for the configuration).  This string is
       then used by ``setuptools_scm`` to produce the correct version,
       based on the git tag, when users install from the tarball.
       However, if there is a branch pointed at the tagged commit,
       then the branch name will also be included in the tarball.
       When the branch eventually moves, anyone who checked the hash
       of the tarball before the branch moved will have an incorrect
       hash.

       To generate the file that GitHub does use::

          git archive v3.7.0 -o matplotlib-3.7.0.tar.gz --prefix=matplotlib-3.7.0/

.. _git archive: https://git-scm.com/docs/git-archive
.. _setuptools_scm: https://github.com/pypa/setuptools_scm

If this is a final release, also create a 'doc' branch (this is not
done for pre-releases)::

   git branch v3.7.0-doc
   git push DANGER v3.7.0-doc

Update (or create) the ``v3.7-doc`` milestone.
The description should include the instruction for meeseeksmachine to backport changes
with the ``v3.7-doc`` milestone to both the ``v3.7.x`` branch and the ``v3.7.0-doc`` branch::

   Documentation changes (.rst files and examples)

   on-merge: backport to v3.7.x
   on-merge: backport to v3.7.0-doc

Check all active milestones for consistency. Older doc milestones should also backport to
higher minor versions (e.g. ``v3.6-doc`` should backport to both ``v3.6.x`` and ``v3.7.x``
if the ``v3.7.x`` branch exists)


.. _release_DOI:

Release management / DOI
========================

Via the `GitHub UI <https://github.com/matplotlib/matplotlib/releases>`__, turn the
newly pushed tag into a release. If this is a pre-release remember to mark it as such.

For final releases, also get the DOI from `Zenodo <https://zenodo.org/>`__ (which will
automatically produce one once the tag is pushed). Add the DOI post-fix and version to
the dictionary in :file:`tools/cache_zenodo_svg.py` and run the script.

This will download the new SVG to :file:`doc/_static/zenodo_cache/{postfix}.svg` and
edit :file:`doc/users/project/citing.rst`. Commit the new SVG, the change to
:file:`tools/cache_zenodo_svg.py`, and the changes to :file:`doc/users/project/citing.rst`
to the VER-doc branch and push to GitHub. ::

  git checkout v3.7.0-doc
  $EDITOR tools/cache_zenodo_svg.py
  python tools/cache_zenodo_svg.py
  git commit -a
  git push DANGER v3.7.0-doc:v3.7.0-doc


.. _release_bld_bin:

Building binaries
=================

We distribute macOS, Windows, and many Linux wheels as well as a source tarball via
PyPI. Most builders should trigger automatically once the tag is pushed to GitHub:

* Windows, macOS and manylinux wheels are built on GitHub Actions. Builds are triggered
  by the GitHub Action defined in :file:`.github/workflows/cibuildwheel.yml`, and wheels
  will be available as artifacts of the build.
* The auto-tick bot should open a pull request into the `conda-forge feedstock
  <https://github.com/conda-forge/matplotlib-feedstock>`__. Review and merge (if you
  have the power to).

.. warning::

   Because this is automated, it is extremely important to bump all branches away from
   the tag as discussed in :ref:`release_tag`.


.. _release_upload_bin:

Make distribution and upload to PyPI
====================================

Once you have collected all of the wheels (expect this to take a few hours), generate
the tarball::

  git checkout v3.7.0
  git clean -xfd
  python -m build --sdist

and copy all of the wheels into :file:`dist` directory. First, check that the dist files
are OK::

  twine check dist/*

and then use ``twine`` to upload all of the files to PyPI ::

   twine upload -s dist/matplotlib*tar.gz
   twine upload dist/*whl

Congratulations, you have now done the second scariest part!


.. _release_docs:

Build and deploy documentation
==============================

To build the documentation you must have the tagged version installed, but
build the docs from the ``ver-doc`` branch.  An easy way to arrange this is::

  pip install matplotlib
  pip install -r requirements/doc/doc-requirements.txt
  git checkout v3.7.0-doc
  git clean -xfd
  make -Cdoc O="-t release -j$(nproc)" html latexpdf LATEXMKOPTS="-silent -f"

which will build both the HTML and PDF version of the documentation.

The built documentation exists in the `matplotlib.github.com
<https://github.com/matplotlib/matplotlib.github.com/>`__ repository.
Pushing changes to main automatically updates the website.

The documentation is organized in subdirectories by version. The latest stable release
is symlinked from the :file:`stable` directory. The documentation for current main is
built on Circle CI and pushed to the `devdocs
<https://github.com/matplotlib/devdocs/>`__ repository. These are available at
`matplotlib.org/devdocs <https://matplotlib.org/devdocs/>`__.

Assuming you have this repository checked out in the same directory as
matplotlib ::

  cd ../matplotlib.github.com
  cp -a ../matplotlib/doc/build/html 3.7.0
  rm 3.7.0/.buildinfo
  cp ../matplotlib/doc/build/latex/Matplotlib.pdf 3.7.0

which will copy the built docs over.  If this is a final release, link the
``stable`` subdirectory to the newest version::

  rm stable
  ln -s 3.7.0 stable

You will need to manually edit :file:`versions.html` to show the released version.
You will also need to edit :file:`sitemap.xml` to include
the newly released version.  Now commit and push everything to GitHub ::

  git add *
  git commit -a -m 'Updating docs for v3.7.0'
  git push DANGER main

Congratulations you have now done the third scariest part!

If you have access, clear the CloudFlare caches.

It typically takes about 5-10 minutes for the website to process the push and update the
live web page (remember to clear your browser cache).

.. _release_merge_up:

Merge up changes to main
========================

After a release is done, the changes from the release branch should be merged into the
``main`` branch. This is primarily done so that the released tag is on the main branch
so ``git describe`` (and thus ``setuptools-scm``) has the most current tag.
Secondarily, changes made during release (including removing individualized release
notes, fixing broken links, and updating the version switcher) are bubbled up to
``main``.

Git conflicts are very likely to arise, though aside from changes made directly to the
release branch (mostly as part of the release), they should be relatively-easily resolved
by using the version from ``main``. This is not a universal rule, and care should be
taken to ensure correctness::

   git switch main
   git pull
   git switch -c merge_up_v3.7.0
   git merge v3.7.x
   # resolve conflicts
   git merge --continue

Due to branch protections for the ``main`` branch, this is merged via a standard pull
request, though the PR cleanliness status check is expected to fail. The PR should not
be squashed because the intent is to merge the branch histories.

Publicize this release
======================

After the release is published to PyPI and conda, it should be announced
through our communication channels:

.. rst-class:: checklist

* Send a short version of the release notes and acknowledgments to all the :ref:`mailing-lists`
* Post highlights and link to :ref:`What's new <release-notes>` on the
  active :ref:`social media accounts <social-media>`
* Add a release announcement to the  "News" section of
  `matplotlib.org <https://github.com/matplotlib/mpl-brochure-site>`_ by editing
  ``docs/body.html``. Link to the auto-generated announcement discourse post,
  which is in `Announcements > matplotlib-announcements <https://discourse.matplotlib.org/c/announce/matplotlib-announce/10>`_.

Conda packages
==============

The Matplotlib project itself does not release conda packages. In particular,
the Matplotlib release manager is not responsible for conda packaging.

For information on the packaging of Matplotlib for conda-forge see
https://github.com/conda-forge/matplotlib-feedstock.
