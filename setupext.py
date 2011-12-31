"""
Some helper functions for building the C extensions

You may need to use the "basedirlist" option in setup.cfg to point
to the location of your required libs, eg, png, z, freetype,
overriding the settings hard-coded in the "basedir" directory
below.

DARWIN

  I have installed all of the backends on OSX.

  Tk: If you want to install TkAgg, I recommend the "batteries included"
  binary build of Tcl/Tk at
  http://www.apple.com/downloads/macosx/unix_open_source/tcltkaqua.html

  GTK: I installed GTK from src as described at
  http://www.macgimp.org/index.php?topic=gtk.  There are several
  packages, but all configure/make/make install w/o problem.  In
  addition to the packages listed there, You will also need libpng,
  libjpeg, and libtiff if you want output to these formats from GTK.

WIN32 - MINGW

  If you are sufficiently masochistic that you want to build this
  yourself, download the win32_static dir from
  http://matplotlib.sourceforge.net/win32_static.tar.gz and
  see the README file in that dir

  > python setup.py build --compiler=mingw32 bdist_wininst  > build23.out

  NOTE, if you are building on python24 on win32, see
  http://mail.python.org/pipermail/python-list/2004-December/254826.html

WIN32 - VISUAL STUDIO 7.1 (2003)

  This build is similar to the mingw.  Download the visual studio static
  dependencies from
  http://matplotlib.sourceforge.net/win32_static_vs.tar.gz and
  see the README in that dir

  > python setup.py build --compiler=msvc bdist_wininst

"""

import os
import re
import subprocess
from distutils import sysconfig, version

basedir = {
    'win32'  : ['win32_static',],
    'linux2-alpha' : ['/usr/local', '/usr'],
    'linux2-hppa' : ['/usr/local', '/usr'],
    'linux2-mips' : ['/usr/local', '/usr'],
    'linux2-sparc' : ['/usr/local', '/usr'],
    'linux2' : ['/usr/local', '/usr'],
    'linux3' : ['/usr/local', '/usr'],
    'linux'  : ['/usr/local', '/usr',],
    'cygwin' : ['/usr/local', '/usr',],
    '_darwin' : ['/sw/lib/freetype2', '/sw/lib/freetype219', '/usr/local',
                '/usr', '/sw'],
    # it appears builds with darwin are broken because of all the
    # different flags the deps can be compile with, so I am pushing
    # people to :
    #   make -f make.osx fetch deps mpl_build mpl_install

    'darwin' : [],

    'freebsd4' : ['/usr/local', '/usr'],
    'freebsd5' : ['/usr/local', '/usr'],
    'freebsd6' : ['/usr/local', '/usr'],
    'sunos5' : [os.getenv('MPLIB_BASE') or '/usr/local',],
    'gnukfreebsd5' : ['/usr/local', '/usr'],
    'gnukfreebsd6' : ['/usr/local', '/usr'],
    'gnukfreebsd7' : ['/usr/local', '/usr'],
    'gnukfreebsd8' : ['/usr/local', '/usr'],
    'gnu0' : ['/usr'],
    'aix5' : ['/usr/local'],
}

import sys, os, stat

from textwrap import fill
from distutils.core import Extension
import glob

if sys.version_info[0] < 3:
    import ConfigParser as configparser
    from cStringIO import StringIO
    if sys.platform != 'win32':
        from commands import getstatusoutput
else:
    import configparser
    from io import StringIO
    if sys.platform != 'win32':
        from subprocess import getstatusoutput

BUILT_PNG       = False
BUILT_AGG       = False
BUILT_FT2FONT   = False
BUILT_TTCONV    = False
BUILT_GTKAGG    = False
BUILT_IMAGE     = False
BUILT_MACOSX    = False
BUILT_TKAGG     = False
BUILT_WINDOWING = False
BUILT_CONTOUR   = False
BUILT_DELAUNAY  = False
BUILT_NXUTILS   = False
BUILT_CONTOUR   = False
BUILT_GDK       = False
BUILT_PATH      = False
BUILT_TRI       = False

AGG_VERSION = 'agg24'
TCL_TK_CACHE = None

# for nonstandard installation/build with --prefix variable
numpy_inc_dirs = []

# matplotlib build options, which can be altered using setup.cfg
options = {'display_status': True,
           'verbose': False,
           'provide_pytz': 'auto',
           'provide_dateutil': 'auto',
           'build_agg': True,
           'build_gtk': 'auto',
           'build_gtkagg': 'auto',
           'build_tkagg': 'auto',
           'build_macosx': 'auto',
           'build_image': True,
           'build_windowing': True,
           'backend': None,
           'basedirlist': None}

defines = [
        ('PY_ARRAY_UNIQUE_SYMBOL', 'MPL_ARRAY_API'),
        ('PYCXX_ISO_CPP_LIB', '1')]

setup_cfg = os.environ.get('MPLSETUPCFG', 'setup.cfg')
# Based on the contents of setup.cfg, determine the build options
if os.path.exists(setup_cfg):
    config = configparser.SafeConfigParser()
    config.read(setup_cfg)

    try: options['display_status'] = not config.getboolean("status", "suppress")
    except: pass

    try: options['verbose'] = not config.getboolean("status", "verbose")
    except: pass

    try: options['provide_pytz'] = config.getboolean("provide_packages", "pytz")
    except: options['provide_pytz'] = 'auto'

    try: options['provide_dateutil'] = config.getboolean("provide_packages",
                                                         "dateutil")
    except: options['provide_dateutil'] = 'auto'

    try: options['build_gtk'] = config.getboolean("gui_support", "gtk")
    except: options['build_gtk'] = 'auto'

    try: options['build_gtkagg'] = config.getboolean("gui_support", "gtkagg")
    except: options['build_gtkagg'] = 'auto'

    try: options['build_tkagg'] = config.getboolean("gui_support", "tkagg")
    except: options['build_tkagg'] = 'auto'

    try: options['build_macosx'] = config.getboolean("gui_support", "macosx")
    except: options['build_macosx'] = 'auto'

    try: options['backend'] = config.get("rc_options", "backend")
    except: pass

    try: options['basedirlist'] = config.get("directories", "basedirlist")
    except: pass

# For get_base_flags:
if options['basedirlist']:
    basedirlist = options['basedirlist'].split()
else:
    basedirlist = basedir[sys.platform]
print("basedirlist is: %s" % basedirlist)

if options['display_status']:
    def print_line(char='='):
        print(char * 76)

    def print_status(package, status):
        initial_indent = "%22s: " % package
        indent = ' ' * 24
        print(fill(str(status), width=76,
                   initial_indent=initial_indent,
                   subsequent_indent=indent))

    def print_message(message):
        indent = ' ' * 24 + "* "
        print(fill(str(message), width=76,
                   initial_indent=indent,
                   subsequent_indent=indent))

    def print_raw(section):
        print(section)
else:
    def print_line(*args, **kwargs):
        pass
    print_status = print_message = print_raw = print_line

# Remove the -Wstrict-prototypesoption, is it's not valid for C++
customize_compiler = sysconfig.customize_compiler
def my_customize_compiler(compiler):
    retval = customize_compiler(compiler)
    try:
        compiler.compiler_so.remove('-Wstrict-prototypes')
    except (ValueError, AttributeError):
        pass
    return retval

sysconfig.customize_compiler = my_customize_compiler



def run_child_process(cmd):
    p = subprocess.Popen(cmd, shell=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         close_fds=(sys.platform != 'win32'))
    return p.stdin, p.stdout

class CleanUpFile:
    """CleanUpFile deletes the specified filename when self is destroyed."""
    def __init__(self, name):
        self.name = name
    def __del__(self):
        os.remove(self.name)

def temp_copy(_from, _to):
    """temp_copy copies a named file into a named temporary file.
    The temporary will be deleted when the setupext module is destructed.
    """
    # Copy the file data from _from to _to
    s = open(_from).read()
    open(_to,"w+").write(s)
    # Suppress object rebuild by preserving time stamps.
    stats = os.stat(_from)
    os.utime(_to, (stats.st_atime, stats.st_mtime))
    # Make an object to eliminate the temporary file at exit time.
    globals()["_cleanup_"+_to] = CleanUpFile(_to)

def get_win32_compiler():
    # Used to determine mingw32 or msvc
    # This is pretty bad logic, someone know a better way?
    for v in sys.argv:
        if 'mingw32' in v:
            return 'mingw32'
    return 'msvc'
win32_compiler = get_win32_compiler()
if sys.platform == 'win32' and win32_compiler == 'msvc':
    std_libs = []
else:
    std_libs = ['stdc++', 'm']

def has_pkgconfig():
    if has_pkgconfig.cache is not None:
        return has_pkgconfig.cache
    if sys.platform == 'win32':
        has_pkgconfig.cache = False
    else:
        #print 'environ',  os.environ['PKG_CONFIG_PATH']
        status, output = getstatusoutput("pkg-config --help")
        has_pkgconfig.cache = (status == 0)
    return has_pkgconfig.cache
has_pkgconfig.cache = None

def get_pkgconfig(module,
                  packages,
                  flags="--libs --cflags",
                  pkg_config_exec='pkg-config',
                  report_error=False):
    """Loosely based on an article in the Python Cookbook:
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/502261"""
    if not has_pkgconfig():
        return False

    _flags = {'-I': 'include_dirs',
              '-L': 'library_dirs',
              '-l': 'libraries',
              '-D': 'define_macros',
              '-U': 'undef_macros'}

    cmd = "%s %s %s" % (pkg_config_exec, flags, packages)
    status, output = getstatusoutput(cmd)
    if status == 0:
        for token in output.split():
            attr = _flags.get(token[:2], None)
            if attr is not None:
                if token[:2] == '-D':
                    value = tuple(token[2:].split('='))
                    if len(value) == 1:
                        value = (value[0], None)
                else:
                    value = token[2:]
                set = getattr(module, attr)
                if value not in set:
                    set.append(value)
            else:
                if token not in module.extra_link_args:
                    module.extra_link_args.append(token)
        return True
    if report_error:
        print_status("pkg-config", "looking for %s" % packages)
        print_message(output)
    return False

def get_pkgconfig_version(package):
    default = "found, but unknown version (no pkg-config)"
    if not has_pkgconfig():
        return default

    status, output = getstatusoutput(
        "pkg-config %s --modversion" % (package))
    if status == 0:
        return output
    return default

def try_pkgconfig(module, package, fallback):
    if not get_pkgconfig(module, package):
        module.libraries.append(fallback)

def find_include_file(include_dirs, filename):
    for d in include_dirs:
        if os.path.exists(os.path.join(d, filename)):
            return True
    return False

def check_for_freetype():
    module = Extension('test', [])
    add_base_flags(module)
    if not get_pkgconfig(module, 'freetype2'):
        basedirs = module.include_dirs[:]  # copy the list to avoid inf loop!
        for d in basedirs:
            module.include_dirs.append(os.path.join(d, 'freetype2'))

    print_status("freetype2", get_pkgconfig_version('freetype2'))
    if not find_include_file(module.include_dirs, 'ft2build.h'):
        print_message(
            "WARNING: Could not find 'freetype2' headers in any of %s." %
            ", ".join(["'%s'" % x for x in module.include_dirs]))

    return True

def check_for_libpng():
    module = Extension("test", [])
    get_pkgconfig(module, 'libpng')
    add_base_flags(module)

    print_status("libpng", get_pkgconfig_version('libpng'))
    if not find_include_file(module.include_dirs, 'png.h'):
        print_message(
            "Could not find 'libpng' headers in any of %s" %
            ", ".join(["'%s'" % x for x in module.include_dirs]))

    return True

def add_base_flags(module):
    incdirs = filter(os.path.exists,
                     [os.path.join(p, 'include') for p in basedirlist ])
    libdirs = filter(os.path.exists,
                     [os.path.join(p, 'lib')     for p in basedirlist ]+
                     [os.path.join(p, 'lib64')     for p in basedirlist ] )

    module.include_dirs.extend(incdirs)
    module.include_dirs.append('.')
    module.library_dirs.extend(libdirs)

def getoutput(s):
    'get the output of a system command'

    ret =  os.popen(s).read().strip()
    return ret

def convert_qt_version(version):
    version = '%x'%version
    temp = []
    while len(version) > 0:
        version, chunk = version[:-2], version[-2:]
        temp.insert(0, str(int(chunk, 16)))
    return '.'.join(temp)

def check_for_qt():
    try:
        import pyqtconfig
    except ImportError:
        print_status("Qt", "no")
        return False
    else:
        try:
            qt_version = pyqtconfig.Configuration().qt_version
            qt_version = convert_qt_version(qt_version)
        except AttributeError:
            qt_version = "<unknown>"
        print_status("Qt", "Qt: %s, PyQt: %s" %
                     (qt_version,
                      pyqtconfig.Configuration().pyqt_version_str))
        return True

def check_for_qt4():
    try:
        from PyQt4 import pyqtconfig
    except ImportError:
        print_status("Qt4", "no")
        return False
    else:
        print_status("Qt4", "Qt: %s, PyQt4: %s" %
                     (convert_qt_version(pyqtconfig.Configuration().qt_version),
                      pyqtconfig.Configuration().pyqt_version_str))
        return True

def check_for_pyside():
    try:
        from PySide import __version__
        from PySide import QtCore
    except ImportError:
        print_status("PySide", "no")
        return False
    else:
        print_status("PySide", "Qt: %s, PySide: %s" %
                     (QtCore.__version__, __version__))
        return True

def check_for_cairo():
    try:
        import cairo
    except ImportError:
        print_status("Cairo", "no")
        return False
    else:
        print_status("Cairo", cairo.version)
        return True

def check_for_datetime():
    try:
        import datetime
    except ImportError:
        print_status("datetime", "no")
        return False
    else:
        print_status("datetime", "present, version unknown")
        return True

def check_provide_pytz(hasdatetime=True):
    if hasdatetime and (options['provide_pytz'] is True):
        print_status("pytz", "matplotlib will provide")
        return True
    try:
        import pytz
    except ImportError:
        if hasdatetime and options['provide_pytz']:
            print_status("pytz", "matplotlib will provide")
            return True
        else:
            print_status("pytz", "no")
            return False
    else:
        if pytz.__version__.endswith('mpl'):
            print_status("pytz", "matplotlib will provide")
            return True
        else:
            print_status("pytz", pytz.__version__)
            return False

def check_provide_dateutil(hasdatetime=True):
    if hasdatetime and (options['provide_dateutil'] is True):
        print_status("dateutil", "matplotlib will provide")
        return True
    try:
        import dateutil
    except ImportError:
        if hasdatetime and options['provide_dateutil']:
            print_status("dateutil", "matplotlib will provide")
            return True
        else:
            print_status("dateutil", "no")
            return False
    else:
        try:
            if dateutil.__version__.endswith('mpl'):
                print_status("dateutil", "matplotlib will provide")
                return True
            else:
                print_status("dateutil", dateutil.__version__)
                return False
        except AttributeError:
            print_status("dateutil", "present, version unknown")
            return False

def check_for_dvipng():
    try:
        stdin, stdout = run_child_process('dvipng -version')
        print_status("dvipng", stdout.readlines()[1].decode().split()[-1])
        return True
    except (IndexError, ValueError):
        print_status("dvipng", "no")
        return False

def check_for_ghostscript():
    try:
        if sys.platform == 'win32':
            command = 'gswin32c --version'
        else:
            command = 'gs --version'
        stdin, stdout = run_child_process(command)
        print_status("ghostscript", stdout.read().decode()[:-1])
        return True
    except (IndexError, ValueError):
        print_status("ghostscript", "no")
        return False

def check_for_latex():
    try:
        stdin, stdout = run_child_process('latex -version')
        line = stdout.readlines()[0].decode()
        pattern = '(3\.1\d+)|(MiKTeX \d+.\d+)'
        match = re.search(pattern, line)
        print_status("latex", match.group(0))
        return True
    except (IndexError, ValueError, AttributeError):
        print_status("latex", "no")
        return False

def check_for_pdftops():
    try:
        stdin, stdout = run_child_process('pdftops -v')
        for line in stdout.readlines():
            line = line.decode()
            if 'version' in line:
                print_status("pdftops", line.split()[-1])
                return True
    except (IndexError, ValueError):
        print_status("pdftops", "no")
        return False

def check_for_numpy(min_version):
    try:
        import numpy
    except ImportError:
        print_status("numpy", "no")
        print_message("You must install numpy %s or later to build matplotlib." %
                      min_version)
        return False

    expected_version = version.LooseVersion(min_version)
    found_version = version.LooseVersion(numpy.__version__)
    if not found_version >= expected_version:
        print_message(
            'numpy %s or later is required; you have %s' %
            (min_version, numpy.__version__))
        return False
    module = Extension('test', [])
    add_numpy_flags(module)
    add_base_flags(module)

    print_status("numpy", numpy.__version__)
    if not find_include_file(module.include_dirs, os.path.join("numpy", "arrayobject.h")):
        print_message("Could not find the headers for numpy.  You may need to install the development package.")
        return False
    return True

def add_numpy_flags(module):
    "Add the modules flags to build extensions which use numpy"
    import numpy
    module.include_dirs.append(numpy.get_include())

def add_png_flags(module):
    try_pkgconfig(module, 'libpng', 'png')
    add_base_flags(module)
    add_numpy_flags(module)
    module.libraries.append('z')
    module.include_dirs.extend(['.'])
    module.libraries.extend(std_libs)

def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    # before adding the freetype flags since -z comes later
    add_base_flags(module)
    add_numpy_flags(module)
    module.include_dirs.extend(['src', '%s/include'%AGG_VERSION, '.'])

    # put these later for correct link order
    module.libraries.extend(std_libs)

def add_ft2font_flags(module):
    'Add the module flags to ft2font extension'
    add_numpy_flags(module)
    if not get_pkgconfig(module, 'freetype2'):
        module.libraries.extend(['freetype', 'z'])
        add_base_flags(module)

        basedirs = module.include_dirs[:]  # copy the list to avoid inf loop!
        for d in basedirs:
            module.include_dirs.append(os.path.join(d, 'freetype2'))
            p = os.path.join(d, 'lib/freetype2/include')
            if os.path.exists(p): module.include_dirs.append(p)
            p = os.path.join(d, 'lib/freetype2/include/freetype2')
            if os.path.exists(p): module.include_dirs.append(p)

        basedirs = module.library_dirs[:]  # copy the list to avoid inf loop!
        for d in basedirs:
            p = os.path.join(d, 'freetype2/lib')
            if os.path.exists(p): module.library_dirs.append(p)
    else:
        add_base_flags(module)
        module.libraries.append('z')

    # put this last for library link order
    module.libraries.extend(std_libs)

def check_for_gtk():
    'check for the presence of pygtk'
    gotit = False
    explanation = None
    try:
        import gtk
    except ImportError:
        explanation = 'Building for Gtk+ requires pygtk; you must be able to "import gtk" in your build/install environment'
    except RuntimeError:
        explanation = 'pygtk present but import failed'
    else:
        version = (2,2,0)
        if gtk.pygtk_version < version:
            explanation = "Error: GTK backend requires PyGTK %d.%d.%d (or later), " \
                  "%d.%d.%d was detected." % (
                version + gtk.pygtk_version)
        else:
            gotit = True

    if gotit:
        module = Extension('test', [])
        add_pygtk_flags(module)
        if not find_include_file(module.include_dirs, os.path.join("gtk", "gtk.h")):
            explanation = (
                "Could not find Gtk+ headers in any of %s" %
                ", ".join(["'%s'" % x for x in module.include_dirs]))
            gotit = False

    def ver2str(tup):
        return ".".join([str(x) for x in tup])

    if gotit:
        import gobject
        if hasattr(gobject, 'pygobject_version'):
            pygobject_version = ver2str(gobject.pygobject_version)
        else:
            pygobject_version = '[pre-pygobject]'
        print_status("Gtk+", "gtk+: %s, glib: %s, pygtk: %s, pygobject: %s" %
                     (ver2str(gtk.gtk_version), ver2str(gobject.glib_version),
                      ver2str(gtk.pygtk_version), pygobject_version))
    else:
        print_status("Gtk+", "no")

    if explanation is not None:
        print_message(explanation)

    # Switch off the event loop for PyGTK >= 2.15.0
    if gotit:
        try:
            gtk.set_interactive(False)
        except AttributeError: # PyGTK < 2.15.0
            pass

    return gotit

def add_pygtk_flags(module):
    'Add the module flags to build extensions which use gtk'

    if sys.platform=='win32':
        # popen broken on my win32 plaform so I can't use pkgconfig
        module.library_dirs.extend(
            ['C:/GTK/bin', 'C:/GTK/lib'])

        module.include_dirs.extend(
            ['win32_static/include/pygtk-2.0',
             'C:/GTK/include',
             'C:/GTK/include/gobject',
             'C:/GTK/include/gmodule',
             'C:/GTK/include/glib',
             'C:/GTK/include/pango',
             'C:/GTK/include/atk',
             'C:/GTK/include/X11',
             'C:/GTK/include/cairo',
             'C:/GTK/include/gdk',
             'C:/GTK/include/gdk-pixbuf',
             'C:/GTK/include/gtk',
             ])

        add_base_flags(module)

        if 'PKG_CONFIG_PATH' not in os.environ:
            # If Gtk+ is installed, pkg-config is required to be installed
            os.environ['PKG_CONFIG_PATH'] = 'C:\GTK\lib\pkgconfig'

        pygtkIncludes = getoutput('pkg-config --cflags-only-I pygtk-2.0').split()
        gtkIncludes = getoutput('pkg-config --cflags-only-I gtk+-2.0').split()
        includes = pygtkIncludes + gtkIncludes
        module.include_dirs.extend([include[2:] for include in includes])

        pygtkLinker = getoutput('pkg-config --libs pygtk-2.0').split()
        gtkLinker =  getoutput('pkg-config --libs gtk+-2.0').split()
        linkerFlags = pygtkLinker + gtkLinker

        module.libraries.extend(
            [flag[2:] for flag in linkerFlags if flag.startswith('-l')])

        module.library_dirs.extend(
            [flag[2:] for flag in linkerFlags if flag.startswith('-L')])

        module.extra_link_args.extend(
            [flag for flag in linkerFlags if not
             (flag.startswith('-l') or flag.startswith('-L'))])

        # visual studio doesn't need the math library
        if sys.platform == 'win32' and win32_compiler == 'msvc' and 'm' in module.libraries:
            module.libraries.remove('m')

    if sys.platform != 'win32':
        # If Gtk+ is installed, pkg-config is required to be installed
        add_base_flags(module)
        ok = get_pkgconfig(module, 'pygtk-2.0 gtk+-2.0', report_error=True)
        if not ok:
            print_message(
                "You may need to install 'dev' package(s) to provide header files.")
    # visual studio doesn't need the math library
    if sys.platform == 'win32' and win32_compiler == 'msvc' and 'm' in module.libraries:
        module.libraries.remove('m')

# Make sure you use the Tk version given by Tkinter.TkVersion
# or else you'll build for a wrong version of the Tcl
# interpreter (leading to nasty segfaults).
def check_for_tk():
    gotit = False
    explanation = None
    try:
        import Tkinter
    except ImportError:
        explanation = 'TKAgg requires Tkinter'
    except RuntimeError:
        explanation = 'Tkinter present but import failed'
    else:
        if Tkinter.TkVersion < 8.3:
            explanation = "Tcl/Tk v8.3 or later required"
        else:
            gotit = True

    if gotit:
        module = Extension('test', [])
        try:
            explanation = add_tk_flags(module)
        except RuntimeError:
            # This deals with the change in exception handling syntax in
            # python 3. If we only need to support >= 2.6, we can just use the
            # commented out lines below.
            exc_type,exc,tb = sys.exc_info()
            explanation = str(exc)
            gotit = False
#        except RuntimeError, e:
#            explanation = str(e)
        else:
            if not find_include_file(module.include_dirs, "tk.h"):
                message = 'Tkinter present, but header files are not found. ' + \
                          'You may need to install development packages.'
                if explanation is not None:
                    explanation += '\n' + message
                else:
                    explanation = message
                gotit = False

    if gotit:
        try:
            tk_v = Tkinter.__version__.split()[-2]
        except (AttributeError, IndexError):
            # Tkinter.__version__ has been removed in python 3
            tk_v = 'version not identified'
        print_status("Tkinter", "Tkinter: %s, Tk: %s, Tcl: %s" %
                     (tk_v, Tkinter.TkVersion, Tkinter.TclVersion))
    else:
        print_status("Tkinter", "no")
    if explanation is not None:
        print_message(explanation)
    return gotit

def check_for_macosx():
    gotit = False
    import sys
    if sys.platform=='darwin':
        gotit = True
    if gotit:
        print_status("Mac OS X native", "yes")
    else:
        print_status("Mac OS X native", "no")
    return gotit

def query_tcltk():
    """Tries to open a Tk window in order to query the Tk object about its library paths.
       This should never be called more than once by the same process, as Tk intricacies
       may cause the Python interpreter to hang. The function also has a workaround if
       no X server is running (useful for autobuild systems)."""
    global TCL_TK_CACHE
    # Use cached values if they exist, which ensures this function only executes once
    if TCL_TK_CACHE is not None:
        return TCL_TK_CACHE

    # By this point, we already know that Tkinter imports correctly
    import Tkinter
    tcl_lib_dir = ''
    tk_lib_dir = ''
    # First try to open a Tk window (requires a running X server)
    try:
        tk = Tkinter.Tk()
    except Tkinter.TclError:
        # Next, start Tcl interpreter without opening a Tk window (no need for X server)
        # This feature is available in python version 2.4 and up
        try:
            tcl = Tkinter.Tcl()
        except AttributeError:    # Python version not high enough
            pass
        except Tkinter.TclError:  # Something went wrong while opening Tcl
            pass
        else:
            tcl_lib_dir = str(tcl.getvar('tcl_library'))
            # Guess Tk location based on Tcl location
            (head, tail) = os.path.split(tcl_lib_dir)
            tail = tail.replace('Tcl', 'Tk').replace('tcl', 'tk')
            tk_lib_dir = os.path.join(head, tail)
            if not os.path.exists(tk_lib_dir):
                tk_lib_dir = tcl_lib_dir.replace('Tcl', 'Tk').replace('tcl', 'tk')
    else:
        # Obtain Tcl and Tk locations from Tk widget
        tk.withdraw()
        tcl_lib_dir = str(tk.getvar('tcl_library'))
        tk_lib_dir = str(tk.getvar('tk_library'))
        tk.destroy()

    # Save directories and version string to cache
    TCL_TK_CACHE = tcl_lib_dir, tk_lib_dir, str(Tkinter.TkVersion)[:3]
    return TCL_TK_CACHE

def parse_tcl_config(tcl_lib_dir, tk_lib_dir):
    import Tkinter
    tcl_poss = [tcl_lib_dir,
                os.path.normpath(os.path.join(tcl_lib_dir, '..')),
                "/usr/lib/tcl"+str(Tkinter.TclVersion),
                "/usr/lib"]
    tk_poss = [tk_lib_dir,
                os.path.normpath(os.path.join(tk_lib_dir, '..')),
               "/usr/lib/tk"+str(Tkinter.TkVersion),
               "/usr/lib"]
    for ptcl, ptk in zip(tcl_poss, tk_poss):
        tcl_config = os.path.join(ptcl, "tclConfig.sh")
        tk_config = os.path.join(ptk, "tkConfig.sh")
        if (os.path.exists(tcl_config) and os.path.exists(tk_config)):
            break
    if not (os.path.exists(tcl_config) and os.path.exists(tk_config)):
        return None

    def get_var(file, varname):
        p = subprocess.Popen(
            '. %s ; eval echo ${%s}' % (file, varname),
            shell=True,
            executable="/bin/sh",
            stdout=subprocess.PIPE)
        result = p.communicate()[0]
        return result

    tcl_lib_dir = get_var(tcl_config, 'TCL_LIB_SPEC').split()[0][2:].strip()
    tcl_inc_dir = get_var(tcl_config, 'TCL_INCLUDE_SPEC')[2:].strip()
    tcl_lib = get_var(tcl_config, 'TCL_LIB_FLAG')[2:].strip()

    tk_lib_dir = get_var(tk_config, 'TK_LIB_SPEC').split()[0][2:].strip()
    tk_inc_dir = get_var(tk_config, 'TK_INCLUDE_SPEC').strip()
    if tk_inc_dir == '':
        tk_inc_dir = tcl_inc_dir
    else:
        tk_inc_dir = tk_inc_dir[2:]
    tk_lib = get_var(tk_config, 'TK_LIB_FLAG')[2:].strip()

    if not os.path.exists(os.path.join(tk_inc_dir, 'tk.h')):
        return None

    return tcl_lib_dir, tcl_inc_dir, tcl_lib, tk_lib_dir, tk_inc_dir, tk_lib

def guess_tcl_config(tcl_lib_dir, tk_lib_dir, tk_ver):
    if not (os.path.exists(tcl_lib_dir) and os.path.exists(tk_lib_dir)):
        return None

    tcl_lib = os.path.normpath(os.path.join(tcl_lib_dir, '../'))
    tk_lib = os.path.normpath(os.path.join(tk_lib_dir, '../'))

    tcl_inc = os.path.normpath(os.path.join(tcl_lib_dir,
                                                    '../../include/tcl' + tk_ver))
    if not os.path.exists(tcl_inc):
        tcl_inc = os.path.normpath(os.path.join(tcl_lib_dir,
                                                '../../include'))

    tk_inc = os.path.normpath(os.path.join(tk_lib_dir,
                                           '../../include/tk' + tk_ver))
    if not os.path.exists(tk_inc):
        tk_inc = os.path.normpath(os.path.join(tk_lib_dir,
                                                       '../../include'))

    if not os.path.exists(os.path.join(tk_inc, 'tk.h')):
        tk_inc = tcl_inc

    if not os.path.exists(tcl_inc):
        # this is a hack for suse linux, which is broken
        if (sys.platform.startswith('linux') and
            os.path.exists('/usr/include/tcl.h') and
            os.path.exists('/usr/include/tk.h')):
            tcl_inc = '/usr/include'
            tk_inc = '/usr/include'

    if not os.path.exists(os.path.join(tk_inc, 'tk.h')):
        return None

    return tcl_lib, tcl_inc, 'tcl' + tk_ver, tk_lib, tk_inc, 'tk' + tk_ver

def hardcoded_tcl_config():
    tcl_inc = "/usr/local/include"
    tk_inc = "/usr/local/include"
    tcl_lib = "/usr/local/lib"
    tk_lib = "/usr/local/lib"
    return tcl_lib, tcl_inc, 'tcl', tk_lib, tk_inc, 'tk'

def add_tk_flags(module):
    'Add the module flags to build extensions which use tk'
    message = None
    if sys.platform == 'win32':
        major, minor1, minor2, s, tmp = sys.version_info
        if major == 2 and minor1 in [6, 7]:
            module.include_dirs.extend(['win32_static/include/tcl85'])
            module.libraries.extend(['tk85', 'tcl85'])
        elif major == 2 and minor1 in [3, 4, 5]:
            module.include_dirs.extend(['win32_static/include/tcl84'])
            module.libraries.extend(['tk84', 'tcl84'])
        elif major == 2 and minor1 == 2:
            module.include_dirs.extend(['win32_static/include/tcl83'])
            module.libraries.extend(['tk83', 'tcl83'])
        else:
            raise RuntimeError('No tk/win32 support for this python version yet')
        module.library_dirs.extend([os.path.join(sys.prefix, 'dlls')])

    elif sys.platform == 'darwin':
        # this config section lifted directly from Imaging - thanks to
        # the effbot!

        # First test for a MacOSX/darwin framework install
        from os.path import join, exists
        framework_dirs = [
            join(os.getenv('HOME'), '/Library/Frameworks'),
            '/Library/Frameworks',
            '/System/Library/Frameworks/',
        ]

        # Find the directory that contains the Tcl.framework and Tk.framework
        # bundles.
        # XXX distutils should support -F!
        tk_framework_found = 0
        for F in framework_dirs:
            # both Tcl.framework and Tk.framework should be present
            for fw in 'Tcl', 'Tk':
                if not exists(join(F, fw + '.framework')):
                    break
            else:
                # ok, F is now directory with both frameworks. Continure
                # building
                tk_framework_found = 1
                break
        if tk_framework_found:
            # For 8.4a2, we must add -I options that point inside the Tcl and Tk
            # frameworks. In later release we should hopefully be able to pass
            # the -F option to gcc, which specifies a framework lookup path.
            #
            tk_include_dirs = [
                join(F, fw + '.framework', H)
                for fw in ('Tcl', 'Tk')
                for H in ('Headers', 'Versions/Current/PrivateHeaders')
            ]

            # For 8.4a2, the X11 headers are not included. Rather than include a
            # complicated search, this is a hard-coded path. It could bail out
            # if X11 libs are not found...
            # tk_include_dirs.append('/usr/X11R6/include')
            frameworks = ['-framework', 'Tcl', '-framework', 'Tk']
            module.include_dirs.extend(tk_include_dirs)
            module.extra_link_args.extend(frameworks)
            module.extra_compile_args.extend(frameworks)

    # you're still here? ok we'll try it this way...
    else:
        success = False
        # There are 3 methods to try, in decreasing order of "smartness"
        #
        #   1. Parse the tclConfig.sh and tkConfig.sh files that have
        #      all the information we need
        #
        #   2. Guess the include and lib dirs based on the location of
        #      Tkinter's 'tcl_library' and 'tk_library' variables.
        #
        #   3. Use some hardcoded locations that seem to work on a lot
        #      of distros.

        # Query Tcl/Tk system for library paths and version string
        try:
            tcl_lib_dir, tk_lib_dir, tk_ver = query_tcltk()
        except:
            tk_ver = ''
            result = hardcoded_tcl_config()
        else:
            result = parse_tcl_config(tcl_lib_dir, tk_lib_dir)
            if result is None:
                message = """\
Guessing the library and include directories for Tcl and Tk because the
tclConfig.sh and tkConfig.sh could not be found and/or parsed."""
                result = guess_tcl_config(tcl_lib_dir, tk_lib_dir, tk_ver)
                if result is None:
                    message = """\
Using default library and include directories for Tcl and Tk because a
Tk window failed to open.  You may need to define DISPLAY for Tk to work
so that setup can determine where your libraries are located."""
                    result = hardcoded_tcl_config()

        # Add final versions of directories and libraries to module lists
        tcl_lib_dir, tcl_inc_dir, tcl_lib, tk_lib_dir, tk_inc_dir, tk_lib = result
        module.include_dirs.extend([tcl_inc_dir, tk_inc_dir])
        module.library_dirs.extend([tcl_lib_dir, tk_lib_dir])
        module.libraries.extend([tcl_lib, tk_lib])

    return message

def add_windowing_flags(module):
    'Add the module flags to build extensions using windowing api'
    module.include_dirs.extend(['C:/include'])
    module.libraries.extend(['user32'])
    module.library_dirs.extend(['C:/lib'])
    module.extra_link_args.append("-mwindows")

def build_windowing(ext_modules, packages):
    """windowing is optional and provides functions for managing
       windows better, .e.g.  maintaining focus on win32"""
    global BUILT_WINDOWING
    if BUILT_WINDOWING: return # only build it if you you haven't already
    module = Extension('matplotlib._windowing',
                       ['src/_windowing.cpp'],
                       )
    add_windowing_flags(module)
    ext_modules.append(module)
    BUILT_WINDOWING = True

def build_ft2font(ext_modules, packages):
    global BUILT_FT2FONT
    if BUILT_FT2FONT: return # only build it if you you haven't already
    deps = ['src/ft2font.cpp', 'src/mplutils.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension('matplotlib.ft2font', deps,
                       define_macros=defines)
    add_ft2font_flags(module)
    ext_modules.append(module)
    BUILT_FT2FONT = True

def build_ttconv(ext_modules, packages):
    global BUILT_TTCONV
    if BUILT_TTCONV: return # only build it if you you haven't already
    deps = ['src/_ttconv.cpp',
            'ttconv/pprdrv_tt.cpp',
            'ttconv/pprdrv_tt2.cpp',
            'ttconv/ttutil.cpp']

    module = Extension('matplotlib.ttconv', deps,
                       define_macros=defines)
    add_base_flags(module)
    ext_modules.append(module)
    BUILT_TTCONV = True

def build_gtkagg(ext_modules, packages):
    global BUILT_GTKAGG
    if BUILT_GTKAGG: return # only build it if you you haven't already
    deps = ['src/agg_py_transforms.cpp', 'src/_gtkagg.cpp', 'src/mplutils.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension('matplotlib.backends._gtkagg',
                       deps,
                       define_macros=defines
                       )

    # add agg flags before pygtk because agg only supports freetype1
    # and pygtk includes freetype2.  This is a bit fragile.

    add_agg_flags(module)
    add_ft2font_flags(module)
    add_pygtk_flags(module)
    add_numpy_flags(module)

    ext_modules.append(module)
    BUILT_GTKAGG = True

def build_tkagg(ext_modules, packages):
    global BUILT_TKAGG
    if BUILT_TKAGG: return # only build it if you you haven't already
    deps = ['src/agg_py_transforms.cpp', 'src/_tkagg.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension('matplotlib.backends._tkagg',
                       deps,
                       define_macros=defines
                       )

    add_tk_flags(module) # do this first
    add_agg_flags(module)
    add_ft2font_flags(module)

    ext_modules.append(module)
    BUILT_TKAGG = True


def build_macosx(ext_modules, packages):
    global BUILT_MACOSX
    if BUILT_MACOSX: return # only build it if you you haven't already
    deps = ['src/_macosx.m',
            'CXX/cxx_extensions.cxx',
            'CXX/cxxextensions.c',
            'CXX/cxxsupport.cxx',
            'CXX/IndirectPythonInterface.cxx',
            'src/agg_py_transforms.cpp',
            'src/path_cleanup.cpp']
    module = Extension('matplotlib.backends._macosx',
                       deps,
                       extra_link_args = ['-framework','Cocoa'],
                       define_macros=defines
                      )
    add_numpy_flags(module)
    add_agg_flags(module)
    ext_modules.append(module)
    BUILT_MACOSX = True

def build_png(ext_modules, packages):
    global BUILT_PNG
    if BUILT_PNG: return # only build it if you you haven't already

    deps = ['src/_png.cpp', 'src/mplutils.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension(
        'matplotlib._png',
        deps,
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )

    add_png_flags(module)
    ext_modules.append(module)

    BUILT_PNG = True


def build_agg(ext_modules, packages):
    global BUILT_AGG
    if BUILT_AGG: return # only build it if you you haven't already

    agg = (
           'agg_trans_affine.cpp',
           'agg_bezier_arc.cpp',
           'agg_curves.cpp',
           'agg_vcgen_dash.cpp',
           'agg_vcgen_stroke.cpp',
           'agg_image_filters.cpp',
           )

    deps = ['%s/src/%s'%(AGG_VERSION, name) for name in agg]
    deps.extend(['src/mplutils.cpp', 'src/agg_py_transforms.cpp'])
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))
    temp_copy('src/_backend_agg.cpp', 'src/backend_agg.cpp')
    deps.append('src/backend_agg.cpp')
    module = Extension(
        'matplotlib.backends._backend_agg',
        deps,
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )

    add_numpy_flags(module)
    add_agg_flags(module)
    add_ft2font_flags(module)
    ext_modules.append(module)

    BUILT_AGG = True

def build_path(ext_modules, packages):
    global BUILT_PATH
    if BUILT_PATH: return # only build it if you you haven't already

    agg = (
           'agg_vcgen_contour.cpp',
           'agg_curves.cpp',
           'agg_bezier_arc.cpp',
           'agg_trans_affine.cpp',
           'agg_vcgen_stroke.cpp',
           )

    deps = ['%s/src/%s'%(AGG_VERSION, name) for name in agg]
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    temp_copy('src/_path.cpp', 'src/path.cpp')
    deps.extend(['src/agg_py_transforms.cpp',
                 'src/path_cleanup.cpp',
                 'src/path.cpp'])
    module = Extension(
        'matplotlib._path',
        deps,
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )

    add_numpy_flags(module)

    add_agg_flags(module)
    ext_modules.append(module)

    BUILT_PATH = True

def build_image(ext_modules, packages):
    global BUILT_IMAGE
    if BUILT_IMAGE: return # only build it if you you haven't already

    agg = ('agg_trans_affine.cpp',
           'agg_image_filters.cpp',
           'agg_bezier_arc.cpp',
           )

    temp_copy('src/_image.cpp', 'src/image.cpp')
    deps = ['src/image.cpp', 'src/mplutils.cpp']
    deps.extend(['%s/src/%s'%(AGG_VERSION,name) for name in agg])
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension(
        'matplotlib._image',
        deps,
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )

    add_numpy_flags(module)
    add_agg_flags(module)
    ext_modules.append(module)

    BUILT_IMAGE = True



def build_delaunay(ext_modules, packages):
    global BUILT_DELAUNAY
    if BUILT_DELAUNAY:
        return # only build it if you you haven't already

    sourcefiles=["_delaunay.cpp", "VoronoiDiagramGenerator.cpp",
                 "delaunay_utils.cpp", "natneighbors.cpp"]
    sourcefiles = [os.path.join('lib/matplotlib/delaunay',s) for s in sourcefiles]
    delaunay = Extension('matplotlib._delaunay',sourcefiles,
                         include_dirs=numpy_inc_dirs,
                         define_macros=defines
                         )
    add_numpy_flags(delaunay)
    add_base_flags(delaunay)
    ext_modules.append(delaunay)
    packages.extend(['matplotlib.delaunay'])
    BUILT_DELAUNAY = True


def build_contour(ext_modules, packages):
    global BUILT_CONTOUR
    if BUILT_CONTOUR: return # only build it if you you haven't already

    module = Extension(
        'matplotlib._cntr',
        [ 'src/cntr.c'],
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )
    add_numpy_flags(module)
    add_base_flags(module)
    ext_modules.append(module)

    BUILT_CONTOUR = True


def build_nxutils(ext_modules, packages):
    global BUILT_NXUTILS
    if BUILT_NXUTILS: return # only build it if you you haven't already
    module = Extension(
        'matplotlib.nxutils',
        [ 'src/nxutils.c'],
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )
    add_numpy_flags(module)
    add_base_flags(module)
    ext_modules.append(module)

    BUILT_NXUTILS = True


def build_gdk(ext_modules, packages):
    global BUILT_GDK
    if BUILT_GDK: return # only build it if you you haven't already

    temp_copy('src/_backend_gdk.c', 'src/backend_gdk.c')
    module = Extension(
        'matplotlib.backends._backend_gdk',
        ['src/backend_gdk.c'],
        libraries = [],
        include_dirs=numpy_inc_dirs,
        define_macros=defines
        )

    add_numpy_flags(module)
    add_base_flags(module)
    add_pygtk_flags(module)
    ext_modules.append(module)

    BUILT_GDK = True


def build_tri(ext_modules, packages):
    global BUILT_TRI
    if BUILT_TRI: return # only build it if you you haven't already

    deps = ['lib/matplotlib/tri/_tri.cpp', 'src/mplutils.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension('matplotlib._tri', deps,
                       define_macros=defines)
    add_numpy_flags(module)
    add_base_flags(module)
    ext_modules.append(module)
    BUILT_TRI = True
