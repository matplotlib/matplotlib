.. _release-guide:

*************************
Doing a matplolib release
*************************

A guide for developers who are doing a matplotlib release

* Edit :file:`__init__.py` and bump the version number



When doing a release

.. _release-testing:

Testing
=======

* Run all of the regression tests by running the `tests.py` script at
  the root of the source tree.

* Run :file:`unit/memleak_hawaii3.py` and make sure there are no
  memory leaks

* try some GUI examples, eg :file:`simple_plot.py` with GTKAgg, TkAgg, etc...

* remove font cache and tex cache from :file:`.matplotlib` and test
  with and without cache on some example script

* Optionally, make sure :file:`examples/tests/backend_driver.py` runs
  without errors and check the output of the PNG, PDF, PS and SVG
  backends

.. _release-branching:

Branching
============

Once all the tests are passing and you are ready to do a release, you
need to create a release branch::

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

    git tag -a v1.0.1 a9f3f3a50745

Tags allow developers to quickly checkout different releases by name,
and also provides source download via zip and tarball on github.

.. _release-packaging:

Packaging
=========


* Make sure the :file:`MANIFEST.in` us up to date and remove
  :file:`MANIFEST` so it will be rebuilt by MANIFEST.in

* run `git clean` in the mpl git directory before building the sdist

* unpack the sdist and make sure you can build from that directory

* Use :file:`setup.cfg` to set the default backends.  For windows and
  OSX, the default backend should be TkAgg.  You should also turn on
  or off any platform specific build options you need.  Importantly,
  you also need to make sure that you delete the :file:`build` dir
  after any changes to :file:`setup.cfg` before rebuilding since cruft
  in the :file:`build` dir can get carried along.

* on windows, unix2dos the rc file

* We have a Makefile for the OS X builds in the mpl source dir
  :file:`release/osx`, so use this to prepare the OS X releases.

* We have a Makefile for the win32 mingw builds in the mpl source dir
  :file:`release/win32` which you can use this to prepare the windows
  releases, but this is currently broken for python2.6 as described at
  http://www.nabble.com/binary-installers-for-python2.6--libpng-segfault%2C-MSVCR90.DLL-and-%09mingw-td23971661.html

.. _release-candidate-testing:

Release candidate testing
=========================

Post the release candidates tarballs to the `matplotlib download page
<https://github.com/matplotlib/matplotlib/downloads>`_.  If you have
developer rights, you should see an "Upload a new file" section
there.

.. _release-announcing:

Announcing
==========

Announce the release on matplotlib-announce, matplotlib-users and
matplotlib-devel.  Include a summary of highlights from the CHANGELOG
and/or post the whole CHANGELOG since the last release.
