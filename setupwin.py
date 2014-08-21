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
DEPSDIR = os.path.join(os.path.dirname(os.path.normpath(__file__)), 'windeps')
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

def config_dir():
    segment = 'msvcr{}-x{}'.format('100' if VS2010 else '90', '64' if X64 else '32')
    return os.path.join(DEPSDIR, segment)

def tcl_config_dir():
    return os.path.join(config_dir(), 'tcl85', 'include')

def build_tcl():
    inclib = config_dir()
    tcl_inclib = tcl_config_dir()
    if not os.path.exists(tcl_inclib):
        os.makedirs(tcl_inclib)
    tcl_inclib_x11 = os.path.join(tcl_inclib, 'X11')
    if not os.path.exists(tcl_inclib_x11):
        os.makedirs(tcl_inclib_x11)

    distfile = os.path.join(DEPSDIR, 'tcl8513-src.zip')
    compfile = os.path.join(tcl_inclib, 'tcl.h')
    if not os.path.exists(compfile) or os.path.getmtime(distfile) > os.path.getmtime(compfile):
        zip_extract(distfile, DEPSDIR)
        targetdir = os.path.join(DEPSDIR, 'tcl8.5.13')
        headers = glob.glob(os.path.join(targetdir, 'generic', '*.h'))
        for filename in headers:
            shutil.copy(filename, tcl_inclib)

    distfile = os.path.join(DEPSDIR, 'tk8513-src.zip')
    compfile = os.path.join(tcl_inclib, 'tk.h')
    if not os.path.exists(compfile) or os.path.getmtime(distfile) > os.path.getmtime(compfile):
        zip_extract(distfile, DEPSDIR)
        targetdir = os.path.join(DEPSDIR, 'tk8.5.13')
        headers = glob.glob(os.path.join(targetdir, 'generic', '*.h'))
        for filename in headers:
            shutil.copy(filename, tcl_inclib)
        headers = glob.glob(os.path.join(targetdir, 'xlib', 'X11', '*.*'))
        for filename in headers:
            shutil.copy(filename, tcl_inclib_x11)

ZLIB_BUILD_CMD = """\
@ECHO OFF
REM call "%ProgramFiles%\\Microsoft SDKs\\Windows\\v7.0\\Bin\\SetEnv.Cmd" /Release /{xXX} /xp
call "{vcvarsall}" {xXX}

cd /D %ZLIB%
nmake -f win32\\Makefile.msc clean
nmake -f win32\\Makefile.msc
copy /Y /B *.dll %INCLIB%
copy /Y /B *.lib %INCLIB%
copy /Y /B zlib.lib %INCLIB%\\z.lib
copy /Y /B zlib.h %INCLIB%
copy /Y /B zconf.h %INCLIB%
"""

def build_zlib():
    inclib = config_dir()
    if not os.path.exists(inclib):
        os.mkdir(inclib)

    distfile = os.path.join(DEPSDIR, 'zlib128.zip')
    compfile = os.path.join(inclib, 'z.lib')
    if os.path.exists(compfile) and os.path.getmtime(distfile) < os.path.getmtime(compfile):
        # already built
        return

    zip_extract(distfile, DEPSDIR)

    cmdfile = os.path.join(DEPSDIR, 'build_zlib.cmd')
    with open(cmdfile, 'w') as cmd:
        cmd.write(prepare_build_cmd(ZLIB_BUILD_CMD))

    os.environ['INCLIB'] = inclib
    os.environ['ZLIB'] = os.path.join(DEPSDIR, 'zlib-1.2.8')
    os.system(cmdfile)

LIBPNG_BUILD_CMD = """\
@ECHO OFF
REM call "%ProgramFiles%\\Microsoft SDKs\\Windows\\v7.0\\Bin\\SetEnv.Cmd" /Release /{xXX} /xp
call "{vcvarsall}" {xXX}
set CMAKE="cmake.exe"

set BUILDDIR=%LIBPNG%-build
rd /S /Q %BUILDDIR%
%CMAKE% -G"NMake Makefiles" -H%LIBPNG% -B%BUILDDIR% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DZLIB_INCLUDE_DIR=%INCLIB% ^
    -DZLIB_LIBRARY:FILEPATH=%INCLIB%\\zlib.lib ^
    -DPNG_STATIC=ON ^
    -DPNG_SHARED=OFF
copy /Y /B %BUILDDIR%\\pnglibconf.h %INCLIB%
copy /Y /B %LIBPNG%\\png.h %INCLIB%
copy /Y /B %LIBPNG%\\pngconf.h %INCLIB%
cd %BUILDDIR%
nmake -f Makefile
REM It's a static lib -- no *.dll in sight!
REM copy /Y /B *.dll %INCLIB%
copy /Y /B *.lib %INCLIB%
copy /Y /B libpng16_static.lib %INCLIB%\\png.lib
"""

def build_libpng():
    inclib = config_dir()
    if not os.path.exists(inclib):
        os.mkdir(inclib)

    distfile = os.path.join(DEPSDIR, 'libpng-1.6.7.tar.gz')
    compfile = os.path.join(inclib, 'png.lib')
    if os.path.exists(compfile) and os.path.getmtime(distfile) < os.path.getmtime(compfile):
        # already built
        return

    tar_extract(distfile, DEPSDIR)

    cmdfile = os.path.join(DEPSDIR, 'build_libpng.cmd')
    with open(cmdfile, 'w') as cmd:
        cmd.write(prepare_build_cmd(LIBPNG_BUILD_CMD))

    os.environ['INCLIB'] = inclib
    os.environ['LIBPNG'] = os.path.join(DEPSDIR, 'libpng-1.6.7')
    os.system(cmdfile)

FREETYPE_VERSION = '2.4.11'

FREETYPE_BUILD_CMD = """\
@ECHO OFF
REM call "%ProgramFiles%\\Microsoft SDKs\\Windows\\v7.0\\Bin\\SetEnv.Cmd" /Release /{xXX} /xp
call "{vcvarsall}" {xXX}
set MSBUILD=C:\\Windows\\Microsoft.NET\\Framework\\v4.0.30319\\MSBuild.exe

rd /S /Q %FREETYPE%\\objs
%MSBUILD% %FREETYPE%\\builds\\win32\\{vc20xx}\\freetype.sln /t:Clean;Build /p:Configuration="{config}";Platform={WinXX}
xcopy /Y /E /Q %FREETYPE%\\include %INCLIB%
xcopy /Y /E /Q %FREETYPE%\\objs\\win32\\{vc20xx} %INCLIB%
copy /Y /B %FREETYPE%\\objs\\win32\\{vc20xx}\\*.lib %INCLIB%\\freetype.lib
"""

def build_freetype():
    inclib = config_dir()
    if not os.path.exists(inclib):
        os.mkdir(inclib)

    distfile = os.path.join(DEPSDIR, 'ft2411.zip')
    compfile = os.path.join(inclib, 'freetype.lib')
    if os.path.exists(compfile) and os.path.getmtime(distfile) < os.path.getmtime(compfile):
        # already built
        return

    vc = 'vc2010' if VS2010 else 'vc2008'
    WinXX = 'x64' if X64 else 'Win32'

    zip_extract(distfile, DEPSDIR)
    ft_dir = os.path.join(DEPSDIR, 'freetype-2.4.11')
    fixproj(os.path.join(ft_dir, 'builds', 'win32', vc, 'freetype.sln'), WinXX)
    fixproj(os.path.join(ft_dir, 'builds', 'win32', vc, 'freetype.{}'.format('vcxproj' if VS2010 else 'vcproj')), WinXX)

    cmdfile = os.path.join(DEPSDIR, 'build_freetype.cmd')
    with open(cmdfile, 'w') as cmd:
        cmd.write(prepare_build_cmd(FREETYPE_BUILD_CMD, vc20xx=vc, WinXX=WinXX, config='Release' if VS2010 else 'LIB Release'))

    os.environ['INCLIB'] = inclib
    os.environ['FREETYPE'] = ft_dir
    os.system(cmdfile)
