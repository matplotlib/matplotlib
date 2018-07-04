# This file is copied from https://github.com/jbmohler/matplotlib-winbuild
# Only the needed functions were kept.
"""
This file extracts and builds library dependencies libpng, zlib, & freetype2 on
MS Windows.  It also extract tcl/tk for the header files.

There are four possible build targets -- one for each permutation of VS 2008,
2010 and 32/64 bit.  This builds the configuration that matches the Python
install that is executing.

For Python 2.6, 2.7, 3.2:

- VS 2008, 32 bit -- Windows SDK v7.0
- VS 2008, 64 bit -- Windows SDK v7.0

For Python 3.3, 3.4:

- VS 2010, 32 bit -- Windows SDK v7.1
- VS 2010, 64 bit -- Windows SDK v7.1
"""

import platform
import distutils.msvc9compiler as msvc

# Configuration selection & declaration:
X64 = platform.architecture()[0] == '64bit'
xXX = 'x64' if X64 else 'x86'

def prepare_build_cmd(build_cmd, **kwargs):
    VCVARSALL = msvc.find_vcvarsall(10.0)
    if VCVARSALL == None:
        raise RuntimeError('Microsoft VS 2010 required')
    return build_cmd.format(vcvarsall=VCVARSALL, xXX=xXX, **kwargs)
