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

* copy over the latest mpl *.tar.gz tarball to this directory, update
  the MPLVERSION in the Makefile::
  
* build the wininst binary and egg::

    make installers

	The wininst and egg binaries will reside in :file:`matplotlib-VERSION/dist`
