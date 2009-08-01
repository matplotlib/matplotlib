Building binary releases of OS X

Included here is everything to build a binary package installer for OS
X

Dir Contents
-------------

* :file:`bdist_mkpg` - the distutils.extension to build Installer.app
  mpkg installers.  

* :file:`data` - some config files and patches needed for the build

* :file:`*.tar.gz` - the bdist_mkpg, zlib, png, freetype and mpl
  tarballs

* :file:`Makefile` - all the build commands

How to build
--------------

* You need to make sure to unset PKG_CONFIG_PATH to make sure the
  static linking below is respected.  Otherwise the mpl build script
  will dynamically link using the libs from pkgconfig if you have this
  configured on your box::

      unset PKG_CONFIG_PATH

* OPTIONAL: edit :file:`Makefile` so that the *VERSION variables point
  to the latest zlib, png, freetype

* First fetch all the dependencies and patch bdist_mpkg for OSX 10.5.
  You can do this automatically in one step with::

      make fetch_deps

* install the patched bdist_mpkg, that the fetch_deps step just created::

      cd bdist_mpkg-0.4.4
      sudo python setup.py install

* build the dependencies::

      make dependencies

* copy over the latest mpl *.tar.gz tarball to this directory, update
  the MPLVERSION in the Makefile::

      cp /path/to/mpl/matplotlib.0.98.5.tar.gz .

* build the mkpg binary and egg

    make installers

  The mpkg and egg binaries will reside in :file:`matplotlib-VERSION/dist`

Crib sheet
-------------

Build the dependencies::

    cd release/osx/
    unset PKG_CONFIG_PATH
    make fetch_deps
    cd bdist_mpkg-0.4.4
    sudo python setup.py install
    cd ..
    make dependencies

Build the mpl sdist::

    cd ../..
    python setup.py sdist
    mv dist/matplotlib-0.98.6svn.tar.gz release/osx/

Set the version number in the Makefile to 0.98.6svn and build the
installers ::

    cd release/osx
    make installers


