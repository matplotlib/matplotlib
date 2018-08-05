.. highlight:: bash

.. _release-guide:

===============
 Release Guide
===============

A guide for developers who are doing a matplotlib release.

All Releases
============

.. _release-testing:

Testing
-------

We use `travis-ci <https://travis-ci.org/matplotlib/matplotlib>`__ for
continuous integration.  When preparing for a release, the final
tagged commit should be tested locally before it is uploaded::

   pytest -n 8 .


In addition the following two tests should be run and manually inspected::

   python unit/memleak_hawaii3.py
   pushd examples/tests/
   python backend_driver.py
   popd


.. _release_ghstats:

GitHub Stats
------------

We automatically extract GitHub issue, PRs, and authors from GitHub via the API::

  python tools/github_stats.py --since-tag $TAG --project 'matplotlib/matplotlib' --links > doc/users/github_stats.rst

Review and commit changes.  Some issue/PR titles may not be valid rst (the most common issue is
``*`` which is interpreted as unclosed markup).


.. _release_chkdocs:

Check Docs
----------

Before tagging, update the what's new listing in :file:`doc/users/whats_new.rst`
by merging all files in :file:`doc/users/next_whats_new/` coherently. Also,
temporarily comment out the include and toctree glob; re-instate these after a
release. Finally, make sure that the docs build cleanly ::

  make -Cdoc O=-n$(nproc) html latexpdf

After the docs are built, check that all of the links, internal and external, are still
valid.  We use ``linkchecker`` for this, which has not been ported to python3 yet.  You will
need to create a python2 environment with ``requests==2.9.0`` and linkchecker ::

  conda create -p /tmp/lnkchk python=2 requests==2.9.0
  source activate /tmp/lnkchk
  pip install linkchecker
  pushd doc/build/html
  linkchecker index.html --check-extern

Address any issues which may arise.  The internal links are checked on travis, this should only
flag failed external links.

.. _release_tag:

Create release commit and tag
-----------------------------

To create the tag, first create an empty commit with a very terse set of the release notes
in the commit message ::

  git commit --allow-empty

and then create a signed, annotated tag with the same text in the body
message ::

  git tag -a -s v2.0.0

which will prompt you for your gpg key password and an annotation.
For pre releases it is important to follow :pep:`440` so that the
build artifacts will sort correctly in pypi.  Finally, push the tag to GitHub ::

  git push -t DANGER v2.0.0

Congratulations, the scariest part is done!

To prevent issues with any down-stream builders which download the
tarball from GitHub it is important to move all branches away from the commit
with the tag [#]_::

  git commit --allow-empty
  git push DANGER master


.. [#] The tarball that is provided by GitHub is produced using `git
       archive <https://git-scm.com/docs/git-archive>`__.  We use
       `versioneer <https://github.com/warner/python-versioneer>`__
       which uses a format string in
       :file:`lib/matplotlib/_version.py` to have ``git`` insert a
       list of references to exported commit (see
       :file:`.gitattributes` for the configuration).  This string is
       then used by ``versioneer`` to produce the correct version,
       based on the git tag, when users install from the tarball.
       However, if there is a branch pointed at the tagged commit,
       then the branch name will also be included in the tarball.
       When the branch eventually moves, anyone how checked the hash
       of the tarball before the branch moved will have an incorrect
       hash.

       To generate the file that GitHub does use ::

          git archive v2.0.0 -o matplotlib-2.0.0.tar.gz --prefix=matplotlib-2.0.0/


If this is a final release, also create a 'doc' branch (this is not
done for pre-releases)::

   git branch v2.0.0-doc
   git push DANGER v2.0.0-doc

and if this is a major or minor release, also create a bug-fix branch (a
micro release will be cut off of this branch)::

   git branch v2.0.x
   git push DANGER v2.0.x



.. _release_DOI:

Release Management / DOI
------------------------

Via the GitHub UI (chase down link), turn the newly pushed tag into a
release.  If this is a pre-release remember to mark it as such.

For final releases also get a DOI from `zenodo
<https://zenodo.org/>`__ and edit :file:`doc/_templates/citing.html`
with DOI link and commit to the VER-doc branch and push to GitHub ::

  git checkout v2.0.0-doc
  emacs doc/_templates/citing.html
  git push DANGER v2.0.0-doc:v2.0.0-doc

.. _release_bld_bin:

Building binaries
-----------------

We distribute mac, windows, and many linux wheels as well as a source
tarball via pypi.  Before uploading anything, contact the various
builders.  Mac and manylinux wheels are built on travis .  You need to
edit the :file:`.travis.yml` file and push to master of `the build
project <https://github.com/MacPython/matplotlib-wheels>`__.

Update the ``master`` branch (for pre-releases the ``devel`` branch)
of the `conda-forge feedstock
<https://github.com/conda-forge/matplotlib-feedstock>`__ via pull request.

If this is a final release the following downstream packagers should be contacted:

- Debian
- Fedora
- Arch
- Gentoo
- Macports
- Homebrew
- Christoph Gohlke
- Continuum
- Enthought

This can be done ahead of collecting all of the binaries and uploading to pypi.

.. _release_upload_bin:

make distribution and upload to pypi / SF
-----------------------------------------

Once you have collected all of the wheels, generate the tarball ::

  git checkout v2.0.0
  git clean -xfd
  python setup.py sdist

and copy all of the wheels into :file:`dist` directory.  You should use
``twine`` to upload all of the files to pypi ::

   twine upload -s dist/matplotlib*tar.gz
   twine upload dist/*whl

Congratulations, you have now done the second scariest part!

Additionally, for a final release, upload all of the files to sourceforge.

.. _release_docs:

Build and Deploy Documentation
------------------------------

To build the documentation you must have the tagged version installed, but
build the docs from the ``ver-doc`` branch.  An easy way to arrange this is::

  pip install matplotlib
  pip install -r doc-requirements.txt
  git checkout v2.0.0-doc
  git clean -xfd
  cd doc
  make O=-n$(nproc) html latexpdf

which will build both the html and pdf version of the documentation.


The built documentation exists in the `matplotlib.github.com
<https://github.com/matplotlib/matplotlib.github.com/>`__ repository.
Pushing changes to master automatically updates the website.

The documentation is organized by version.  At the root of the tree is
always the documentation for the latest stable release.  Under that,
there are directories containing the documentation for older versions.
The documentation for current master are built on travis and push to
the `devdocs <https://github.com/matplotlib/devdocs/>`__ repository.
These are available at `matplotlib.org/devdocs
<http://matplotlib.org/devdocs>`__.

Assuming you have this repository checked out in the same directory as
matplotlib ::

  cd ../matplotlib.github.com
  mkdir 2.0.0
  rsync -a ../matplotlib/doc/build/html/* 2.0.0
  cp ../matplotlib/doc/build/latex/Matplotlib.pdf 2.0.0

which will copy the built docs over.  If this is a final release, also
replace the top-level docs ::

  rsync -a 2.0.0/* ./

You will need to manually edit :file:`versions.html` to show the last
3 tagged versions.  Now commit and push everything to GitHub ::

  git add *
  git commit -a -m 'Updating docs for v2.0.0'
  git push DANGER master

Congratulations you have now done the third scariest part!

It typically takes about 5-10 minutes for GitHub to process the push
and update the live web page (remember to clear your browser cache).


Announcing
----------

The final step is to announce the release to the world.  A short
version of the release notes along with acknowledgments should be sent to

- matplotlib-user@python.org
- matplotlib-devel@python.org
- matplotlib-announce@python.org

For final releases announcements should also be sent to the
numpy/scipy/jupyter mailing lists and python-announce.

In addition, announcements should be made on social networks (twitter,
g+, FB).  For major release, `NumFOCUS <http://www.numfocus.org/>`__
should be contacted for inclusion in their newsletter and maybe to
have something posted on their blog.
