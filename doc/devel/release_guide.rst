.. _release-guide:

*************************
Doing a matplolib release
*************************

A guide for developers who are doing a matplotlib release

* Edit :file:`__init__.py` and bump the version number


.. _release-testing:

Testing
=======

* Make sure :file:`examples/tests/backend_driver.py` runs without errors
  and check the output of the PNG, PDF, PS and SVG backends

* Run :file:`unit/memleak_hawaii3.py` and make sure there are no
  memory leaks

* Run :file:`unit/nose_tests.py` and make sure all the unit tests are passing

* try some GUI examples, eg :file:`simple_plot.py` with GTKAgg, TkAgg, etc...

* remove font cache and tex cache from :file:`.matplotlib` and test
  with and without cache on some example script

.. _release-packaging:

Packaging
=========

* Make sure the :file:`MANIFEST.in` us up to date and remove
  :file:`MANIFEST` so it will be rebuilt by MANIFEST.in

* run `svn-clean
  <http://svn.collab.net/repos/svn/trunk/contrib/client-side/svn-clean>`_
  from in the mpl svn directory before building the sdist

* unpack the sdist and make sure you can build from that directory

* Use :file:`setup.cfg` to set the default backends.  For windows and
  OSX, the default backend should be TkAgg.  You should also turn on
  or off any platform specific build options you need.  Importantly,
  you also need to make sure that you delete the :file:`build` dir
  after any changes to file:`setup.cfg` before rebuilding since cruft
  in the :file:`build` dir can get carried along.  I will add this to
  the devel release notes,

* on windows, unix2dos the rc file

* make a branch of the svn revision that is released, in case we need
  to do a bug fix release.  Eg, from the top level of the mpl svn
  tree, the one which has "branches", "tags" and "trunk", do::

      > svn copy trunk/matplotlib branches/v0_98_4_maint

.. _release-uploading:

Uploading
=========

* Post the win32 and OS-X binaries for testing and make a request on
  matplotlib-devel for testing.  Pester us if we don't respond


* ftp the source and binaries to the anonymous FTP site::

    mpl> svn-clean
    mpl> python setup.py sdist
    mpl> cd dist/
    dist> sftp jdh2358@frs.sourceforge.net
    Connecting to frs.sourceforge.net...
    sftp> cd uploads
    sftp> ls
    sftp> lls
    matplotlib-0.98.2.tar.gz
    sftp> put matplotlib-0.98.2.tar.gz
    Uploading matplotlib-0.98.2.tar.gz to /incoming/j/jd/jdh2358/uploads/matplotlib-0.98.2.tar.gz

* go https://sourceforge.net/project/admin/?group_id=80706 and do a
  file release.  Click on the "Admin" tab to log in as an admin, and
  then the "File Releases" tab.  Go to the bottom and click "add
  release" and enter the package name but not the version number in
  the "Package Name" box.  You will then be prompted for the "New
  release name" at which point you can add the version number, eg
  somepackage-0.1 and click "Create this release".

  You will then be taken to a fairly self explanatory page where you
  can enter the Change notes, the release notes, and select which
  packages from the incoming ftp archive you want to include in this
  release.  For each binary, you will need to select the platform and
  file type, and when you are done you click on the "notify users who
  are monitoring this package link"


.. _release-announcing:

Announcing
==========

Announce the release on matplotlib-announce, matplotlib-users and
matplotlib-devel.  Include a summary of highlights from the CHANGELOG
and/or post the whole CHANGELOG since the last release.
