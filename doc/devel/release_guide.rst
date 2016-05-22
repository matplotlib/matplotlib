.. _release-guide:

*************
Release Guide
*************

A guide for developers who are doing a matplotlib release.

Development (beta/RC) release checklist
---------------------------------------

- [ ] Testing
- [ ] Create empty REL: commit
- [ ] Create tag
- [ ] Push tags + branches to GH
- [ ] Notify mac/windows/conda builders
- [ ] Merge everything to master
- [ ] announce release

Final release checklist
-----------------------

- [ ] Testing
- [ ] update gh states
- [ ] Create empty REL: commit
- [ ] Create tag
- [ ] Create branches
- [ ] Push tags + branches to GH
- [ ] release / DOI management
- [ ] Notify mac/windows/conda builders
- [ ] Merge everything to master
- [ ] build documentation
- [ ] upload to pypi / sf
- [ ] deploy updated documentation
- [ ] announce release

Details
-------

.. _release-testing:

Testing
=======

We use `travis-ci <https://travis-ci.org/matplotlib/matplotlib>`__ for
continuous integration.  When preparing for a release, the final
tagged commit should be tested locally before it is uploaded::

   python tests.py --processes=8 --process-timeout=300

In addition ::

   python unit/memleak_hawaii3.py

should be run to check for memory leaks.  Optionally, make sure ::

   cd examples/tests/
   python backend_driver.py

runs without errors and check the output of the PNG, PDF, PS and SVG
backends.


.. _release_ghstats:

Github Stats
============

To make sure everyone's hard work gets credited, regenerate the github
stats.  In the project root run ::

  python tools/github_stats.py --since-tag $TAG --project 'matplotlib/matplotlib' --links > doc/users/github_stats.rst


where `$TAG` is the tag of the last major release.  This will generate
stats for all work done since that release.

- [ ] review and commit changes
- [ ] check for valid rst
- [ ] re-add github-stats link

.. _release_tag:

Create Tag
==========

On the tip of the current branch::

  git commit --allow-empty
  git tag -a -s v1.5.0

The commit and tag message should be very terse release notes should look something
like

  REL: vX.Y.Z

  Something brief description

Development releases should be post-fixed with the proper string following
`PEP 440<https://www.python.org/dev/peps/pep-0440/>`__ conventions.

.. _release-branching:

Branching
=========

Once all the tests are passing and you are ready to do a release, you
need to create a release branch.  These only need to be created when
the second part of the version number changes::

   git checkout -b v1.1.x
   git push git@github.com:matplotlib/matplotlib.git v1.1.x


.. _release-packaging:

Packaging
=========

* Make sure the :file:`MANIFEST.in` is up to date and remove
  :file:`MANIFEST` so it will be rebuilt by MANIFEST.in

* run `git clean` in the mpl git directory before building the sdist

* unpack the sdist and make sure you can build from that directory

* Use :file:`setup.cfg` to set the default backends.  For windows and
  OSX, the default backend should be TkAgg.  You should also turn on
  or off any platform specific build options you need.  Importantly,
  you also need to make sure that you delete the :file:`build` dir
  after any changes to :file:`setup.cfg` before rebuilding since cruft
  in the :file:`build` dir can get carried along.

* On windows, unix2dos the rc file.

* We have a Makefile for the OS X builds in the mpl source dir
  :file:`release/osx`, so use this to prepare the OS X releases.

* We have a Makefile for the win32 mingw builds in the mpl source dir
  :file:`release/win32` which you can use this to prepare the windows
  releases.


Update PyPI
===========

This step tells PyPI about the release and uploads a source
tarball. This should only be done with final (non-release-candidate)
releases, since doing so will hide any available stable releases.

You may need to set up your `.pypirc` file as described in the
`distutils register command documentation
<http://docs.python.org/2/distutils/packageindex.html>`_.

Then updating the record on PyPI is as simple as::

    python setup.py register

This will hide any previous releases automatically.

Then, to upload the source tarball::

    rm -rf dist
    python setup.py sdist upload


Build and deploy Documentation
==============================

The built documentation exists in the `matplotlib.github.com
<https://github.com/matplotlib/matplotlib.github.com/>`_ repository.
Pushing changes to master automatically updates the website.

The documentation is organized by version.  At the root of the tree is
always the documentation for the latest stable release.  Under that,
there are directories containing the documentation for older versions.
The documentation for current master are built on travis and push to
the `devdocs <https://github.com/matplotlib/devdocs/>`__ repository.
These are available `matplotlib.org/devdocs
<http://matplotlib.org/devdocs>`__.  There is a symlink directory
with the name of the most recently released version that points to the
root.  With each new release, these directories may need to be
reorganized accordingly.  Any time these version directories are added
or removed, the `versions.html` file (which contains a list of the
available documentation versions for the user) must also be updated.


In the matplotlib source repository, build the documentation::

  cd doc
  python make.py html
  python make.py latex

Then copy the build products into your local checkout of the
`matplotlib.github.com` repository (assuming here to be checked out in
`~/matplotlib.github.com`::

  cp -r build/html/* ~/matplotlib.github.com
  cp build/latex/Matplotlib.pdf ~/matplotlib.github.com

Then, from the `matplotlib.github.com` directory, commit and push the
changes upstream::

  git commit -m "Updating for v1.0.1"
  git push upstream master



Announcing
==========

Announce the release on matplotlib-announce, matplotlib-users, and
matplotlib-devel.  Final (non-release-candidate) versions should also
be announced on python-announce.  Include a summary of highlights from
the CHANGELOG and/or post the whole CHANGELOG since the last release.
