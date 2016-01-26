# This file is coppied from https://github.com/jbmohler/matplotlib-winbuild
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

from __future__ import print_function, absolute_import
import sys
import platform
import os
import glob
import shutil
import zipfile
import tarfile
import distutils.msvc9compiler as msvc

def fixproj(project_file, bit_target):
    """
    :param bit_target: one of 'Win32' or 'x64'
    """
    with open(project_file, 'r') as fd:
        content = '\n'.join(line.strip() for line in fd if line.strip())
    content = content.replace('Win32', bit_target).replace('x64', bit_target)
    with open(project_file, 'w') as fd:
        fd.write(content)

def tar_extract(tar_file, target):
    with tarfile.open(tar_file, 'r:gz') as tgz:
        tgz.extractall(target)

def zip_extract(zip_file, target):
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(target)

# Configuration selection & declaration:
DEPSSRC = os.path.join(os.path.dirname(os.path.normpath(__file__)), 'deps_source')
DEPSBUILD = os.path.join(os.path.dirname(os.path.normpath(__file__)), 'build')
X64 = platform.architecture()[0] == '64bit'
PYVER = sys.version_info[:2]
VS2010 = PYVER >= (3, 3)
# If not VS2010, then use VS2008

VCVARSALL = None

def prepare_build_cmd(build_cmd, **kwargs):
    global VCVARSALL
    if VCVARSALL == None:
        candidate = msvc.find_vcvarsall(10.0 if VS2010 else 9.0)
        if candidate == None:
            raise RuntimeError('Microsoft VS {} required'.format('2010' if VS2010 else '2008'))
        else:
            VCVARSALL = candidate

    return build_cmd.format(vcvarsall=VCVARSALL, xXX='x64' if X64 else 'x86', **kwargs)
