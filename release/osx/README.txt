Building binary releases of OS X

Included here is everything to build a binay package installer for OS
X

Dir Contents
-------------

* :file:`bdist_mkpg` - the distutils.extension to build Installer.app
  mpkg installers.  It is patched from the tarball with
  file:`data/bdist.patch` because 0.4.3 is broken on OS X 10.5.
  Instructions on how to patch and install are below

* :file:`data` - some config files and patches needed for the build

* :file:`*.tar.gz` - the bdist_mkpg, zlib, png, freetype and mpl
  tarballs

* :file:`Makefile` - all the build commands

How to build
--------------

* OPTIONAL: edit :file:`Makefile` so that the *VERSION variables point
  to the latest zlib, png, freetype

* First fetch all the dependencies and patch bdist_mpkg for OSX 10.5.
  You can do this automatically in one step with::

      make fetch_deps

* install the patched bdist_mpkg, that the fetch_deps step just created::

      cd bdist_mpkg-0.4.3
      sudo python setup.py install

* build the dependencies::

      make dependencies

* copy over the latest mpl *.tar.gz tarball to this directory, update
  the MPLVERSION in the Makefile::

      cp /path/to/mpl/matplotlib.0.98.5.tar.gz .

* build the mkpg binary and egg

    make installers

  The mpkg and egg binaries will reside in :file:`matplotlib-VERSION/dist`
