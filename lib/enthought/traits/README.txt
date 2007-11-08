Introduction
------------

'Traits' is a Python package for creating 'manifestly'-typed Python attributes.

Installation
------------

The Traits package is installed using the standard Python 'distutils' package.

Enter the following command in the 'traits-1.0' directory:

   python setup.py install
   
This will perform a normal install of the Traits package into your Python
installation. Refer to the Python 'distutils' documentation for more
installation options.

Download
--------

The Traits package is available as part of the Enthought Tool Suite (ETS), 
available from:

    http://code.enthought.com/ets/
    
To install ETS using Enthought's egg-based 'Enstaller', download and run:

    http://code.enthought.com/enstaller/run_enstaller.py
    
License
-------

The 'traits' package is available under a BSD style license. 

Contact
-------

If you encounter any problems using the 'traits' package, or have any comments
or suggestions about the package, please contact the author:

   David C. Morrill
   dmorrill@enthought.com
   
For discussion of the Traits package, as well as other tools in the Enthought
Tool Suite, use the enthought-dev mailing list:

    https://mail.enthought.com/mailman/listinfo/enthought-dev

    http://dir.gmane.org/gmane.comp.python.enthought.devel

Prerequisites
-------------

The base Traits package should work on any platform supporting Python >= 1.5.2.
   
The user interface capabilities of the traits package require additional
Python packages to be installed.

The UI toolkit backend that is actively maintained is wxPython. To use it,
install a version >= 2.3.3.1 (available from: http://www.wxpython.org).

A UI toolkit backend for Tkinter exists, but is not actively maintained or
tested. If you wish to try Traits with Tkinter, you must also install:

   - Tkinter (usually installed as part of your Python distribution)
   - PMW (Python MegaWidgets) (available from: http://pmw.sourceforge.net)
