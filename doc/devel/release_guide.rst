.. _release-guide:

**************************
Doing a matplotlib release
**************************

A guide for developers who are doing a matplotlib release.

* Edit :file:`__init__.py` and bump the version number

.. _release-testing:

Testing
=======

* Run all of the regression tests by running the `tests.py` script at
  the root of the source tree.

* Run :file:`unit/memleak_hawaii3.py` and make sure there are no
  memory leaks

* try some GUI examples, e.g., :file:`simple_plot.py` with GTKAgg, TkAgg, etc...

* remove font cache and tex cache from :file:`.matplotlib` and test
  with and without cache on some example script

* Optionally, make sure :file:`examples/tests/backend_driver.py` runs
  without errors and check the output of the PNG, PDF, PS and SVG
  backends

.. _release-branching:

Branching
=========

Once all the tests are passing and you are ready to do a release, you
need to create a release branch.  These only need to be created when
the second part of the version number changes::

   git checkout -b v1.1.x
   git push git@github.com:matplotlib/matplotlib.git v1.1.x

On the branch, do any additional testing you want to do, and then build
binaries and source distributions for testing as release candidates.

For each release candidate as well as for the final release version,
please `git tag` the commit you will use for packaging like so::

    git tag -a v1.1.0rc1

The `-a` flag will allow you to write a message about the tag, and
affiliate your name with it. A reasonable tag message would be something
like ``v1.1.0 Release Candidate 1 (September 24, 2011)``. To tag a
release after the fact, just track down the commit hash, and::

    git tag -a v1.0.1rc1 a9f3f3a50745

Tags allow developers to quickly checkout different releases by name,
and also provides source download via zip and tarball on github.

Then push the tags to the main repository::

    git push upstream v1.0.1rc1

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

Posting files
=============

Our current method is for the release manager to collect all of the
binaries from the platform builders and post the files online on
Sourceforge.  It is also possible that those building the binaries
could upload to directly to Sourceforge.  We also post a source
tarball to PyPI, since ``pip`` no longer trusts files downloaded from
other sites.

There are many ways to upload files to Sourceforge (`scp`, `rsync`,
`sftp`, and a web interface) described in `Sourceforge Release File
System documentation
<https://sourceforge.net/apps/trac/sourceforge/wiki/Release%20files%20for%20download>`_.
Below, we will use `sftp`.

1. Create a directory containing all of the release files and `cd` to it.

2. `sftp` to Sourceforge::

     sftp USERNAME@frs.sourceforge.net:/home/frs/project/matplotlib/matplotlib

3. Make a new directory for the release and move to it::

     mkdir matplotlib-1.1.0rc1
     cd matplotlib-1.1.0rc1

4. Upload all of the files in the current directory on your local machine::

     put *

If this release is a final release, the default download for the
matplotlib project should also be updated.  Login to Sourceforge and
visit the `matplotlib files page
<https://sourceforge.net/projects/matplotlib/files/matplotlib/>`_.
Navigate to the tarball of the release you just updated, click on
"Details" icon (it looks like a lower case ``i``), and make it the
default download for all platforms.

There is a list of direct links to downloads on matplotlib's main
website.  This needs to be manually generated and updated every time
new files are posted.

1. Clone the matplotlib documentation repository and `cd` into it::

     git clone git@github.com:matplotlib/matplotlib.github.com.git
     cd matplotlib.github.com

2. Update the list of downloads that you want to display by editing
   the `downloads.txt` file.  Generally, this should contain the last two
   final releases and any active release candidates.

3. Update the downloads webpage by running the `update_downloads.py`
   script.  This script requires `paramiko` (for `sftp` support) and
   `jinja2` for templating.  Both of these dependencies can be
   installed using pip::

     pip install paramiko
     pip install jinja2

   Then update the download page::

     ./update_downloads.py

   You will be prompted for your Sourceforge username and password.

4. Commit the changes and push them up to github::

     git commit -m "Updating download list"
     git push

Update PyPI
===========

Once the tarball has been posted on Sourceforge, you can register a
link to the new release on PyPI.  This should only be done with final
(non-release-candidate) releases, since doing so will hide any
available stable releases.

You may need to set up your `.pypirc` file as described in the
`distutils register command documentation
<http://docs.python.org/2/distutils/packageindex.html>`_.

Then updating the record on PyPI is as simple as::

    python setup.py register

This will hide any previous releases automatically.

Then, to upload the source tarball::

    rm -rf dist
    python setup.py sdist upload

Documentation updates
=====================

The built documentation exists in the `matplotlib.github.com
<https://github.com/matplotlib/matplotlib.github.com/>`_ repository.
Pushing changes to master automatically updates the website.

The documentation is organized by version.  At the root of the tree is
always the documentation for the latest stable release.  Under that,
there are directories containing the documentation for older versions
as well as the bleeding edge release version called `dev` (usually
based on what's on master in the github repository, but it may also
temporarily be a staging area for proposed changes).  There is also a
symlink directory with the name of the most recently released version
that points to the root.  With each new release, these directories may
need to be reorganized accordingly.  Any time these version
directories are added or removed, the `versions.html` file (which
contains a list of the available documentation versions for the user)
must also be updated.

To make sure everyone's hard work gets credited, regenerate the github
stats.  `cd` into the tools directory and run::

  python github_stats.py $TAG > ../doc/users/github_stats.rst

where `$TAG` is the tag of the last major release.  This will generate
stats for all work done since that release.

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
