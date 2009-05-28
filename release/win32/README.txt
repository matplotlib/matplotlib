Building binary releases of WIN32

Included here is everything to build a binary package installer for WIN32 using MinGW

MinGW Requirements
-------------

* Install MinGW using the "Automated MinGW Installer"::

	(tested with MinGW-5.1.4.exe)
	http://sourceforge.net/project/showfiles.php?group_id=2435&package_id=240780


* Install "MSYS Base System"::

	(tested with MSYS-1.0.10.exe)
	http://sourceforge.net/project/showfiles.php?group_id=2435&package_id=24963

* Install wget from "mingwPORT"::

	(tested with wget-1.9.1-mingwPORT.tar.bz2)
	http://sourceforge.net/project/showfiles.php?group_id=2435&package_id=233332
	NOTE: Uncompress and copy the "wget.exe" file to "C:\MingW\bin\"


* Test your installation.  After installing the above, open MSYS and
  check your install by doing::

    > gcc --version
    > g++ --version

  If you don't have g++, try running the mingw exe installer again,
  and you will be prompted for additional compilers to install.
  Select c++ and you are off to the races.

  Make sure setuptools are installed::

    > /c/python26/python
    >>> import setuptools

  If not, grab the latest ez_setup.py and install it::

    > wget http://peak.telecommunity.com/dist/ez_setup.py
    > /c/python26/python ez_setup.py

Dir Contents
-------------

* :file:`data` - some config files and patches needed for the build

* :file:`Makefile` - all the build commands

How to build
--------------

* Edit the variables as needed in :file:`Makefile`

* Open a msys shell from::

	All Programs -> MinGW -> MSYS -> msys

* First fetch all the dependencies::

      make fetch_deps

* build the dependencies::

      make dependencies

* copy over the latest mpl *.tar.gz tarball to this directory.  You
  can create the source distribution file with ::

    > /c/Python26/python sdist --formats=gztar

  and then copy the dist/matplotlib.VERSION.tar.gz file into the
  directory alongside the Makefile.  Update the MPLVERSION in the
  Makefile::

* build the wininst binary and egg::

    make installers

	The wininst and egg binaries will reside in :file:`matplotlib-VERSION/dist`
