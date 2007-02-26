"""
Some helper functions for building the C extensions

you may need to edit basedir to point to the default location of your
required libs, eg, png, z, freetype

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

  > python setup.py build bdist_wininst

"""

import os

basedir = {
    'win32'  : ['win32_static',],
    'linux2' : ['/usr/local', '/usr',],
    'linux'  : ['/usr/local', '/usr',],
    'cygwin' : ['/usr/local', '/usr',],
    'darwin' : ['/sw/lib/freetype2', '/sw/lib/freetype219', '/usr/local',
                '/usr', '/sw'],
    'freebsd4' : ['/usr/local', '/usr'],
    'freebsd5' : ['/usr/local', '/usr'],
    'freebsd6' : ['/usr/local', '/usr'],
    'sunos5' : [os.getenv('MPLIB_BASE') or '/usr/local',],
    'gnukfreebsd5' : ['/usr/local', '/usr'],
    'gnukfreebsd6' : ['/usr/local', '/usr'],
    'aix5' : ['/usr/local'],
}

import sys, os, stat
from distutils.core import Extension
import glob

major, minor1, minor2, s, tmp = sys.version_info
if major<2 or (major==2 and minor1<3):
    True = 1
    False = 0
else:
    True = True
    False = False

BUILT_AGG       = False
BUILT_FT2FONT   = False
BUILT_GTKAGG    = False
BUILT_IMAGE     = False
BUILT_TKAGG     = False
BUILT_WXAGG     = False
BUILT_WINDOWING = False
BUILT_CONTOUR   = False
BUILT_NXUTILS   = False
BUILT_ENTHOUGHT   = False
BUILT_CONTOUR   = False
BUILT_GDK       = False

AGG_VERSION = 'agg23'

# for nonstandard installation/build with --prefix variable
numeric_inc_dirs = []
numarray_inc_dirs = []

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

def add_base_flags(module):

    incdirs = filter(os.path.exists,
                     [os.path.join(p, 'include') for p in basedir[sys.platform] ])
    libdirs = filter(os.path.exists,
                     [os.path.join(p, 'lib')     for p in basedir[sys.platform] ]+
                     [os.path.join(p, 'lib64')     for p in basedir[sys.platform] ] )

    module.include_dirs.extend(incdirs)
    module.include_dirs.append('.')
    module.library_dirs.extend(libdirs)

def getoutput(s):
    'get the output of a system command'

    ret =  os.popen(s).read().strip()
    return ret

def add_numpy_flags(module):
    "Add the modules flags to build extensions which use numpy"
    import numpy
    # TODO: Remove this try statement when it is no longer needed
    try:
        module.include_dirs.append(numpy.get_include())
    except AttributeError:
        module.include_dirs.append(numpy.get_numpy_include())

def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    # before adding the freetype flags since -z comes later
    module.libraries.append('png')
    module.libraries.append('z')
    add_base_flags(module)
    module.include_dirs.extend(['src','swig', '%s/include'%AGG_VERSION, '.'])



    # put these later for correct link order
    module.libraries.extend(std_libs)

def add_gd_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.append('gd')


def add_ft2font_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.extend(['freetype', 'z' ])
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

    if sys.platform == 'win32' and win32_compiler == 'mingw32':
        module.libraries.append('gw32c')

    # put this last for library link order
    module.libraries.extend(std_libs)



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

    # set for msvc compiler if not present
    if not os.environ.has_key('PKG_CONFIG_PATH'):
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


def find_wx_config():
    """If the WX_CONFIG environment variable has been set, returns it value.
    Otherwise, search for `wx-config' in the PATH directories and return the
    first match found.  Failing that, return None.
    """

    wxconfig = os.getenv('WX_CONFIG')
    if wxconfig is not None:
        return wxconfig

    path = os.getenv('PATH') or ''
    for dir in path.split(':'):
        wxconfig = os.path.join(dir, 'wx-config')
        if os.path.exists(wxconfig):
            return wxconfig

    return None


def check_wxpython_headers(wxconfig):
    """Determines if wxPython.h can be found in one of the wxWidgets include
    directories.
    """

    flags = getoutput(wxconfig + ' --cppflags').split()
    incdirs = [os.path.join(p, 'include') for p in basedir[sys.platform]
               if os.path.exists(p)]

    incdirs += [x[2:] for x in flags if x.startswith('-I')]
    header = os.path.join('wx', 'wxPython', 'wxPython.h')

    for d in incdirs:
        if os.path.exists(os.path.join(d, header)):
            return True
    return False

def check_wxpython_broken_macosx104_version(wxconfig):
    """Determines if we're using a broken wxPython installed by Mac OS X 10.4"""
    if sys.platform == 'darwin':
        if wxconfig == '/usr/bin/wx-config':
            version_full = getoutput(wxconfig + ' --version-full')
            if version_full == '2.5.3.1':
                return False
    return True
    
def add_wx_flags(module, wxconfig):
    """
    Add the module flags to build extensions which use wxPython.
    """

    if sys.platform == 'win32': # just added manually
        wxlibs = ['wxexpath', 'wxjpegh', 'wxmsw26uh',
                  'wxmsw26uh_animate', 'wxmsw26uh_gizmos', 'wxmsw26uh_gizmos_xrc',
                  'wxmsw26uh_gl', 'wxmsw26uh_stc', 'wxpngh', 'wxregexuh', 'wxtiffh', 'wxzlibh']
        module.libraries.extend(wxlibs)
        module.libraries.extend(wxlibs)
        return

    def getWX(fmt, *args):
        return getoutput(wxconfig + ' ' + (fmt % args)).split()

    wxFlags = getWX('--cppflags')
    wxLibs = getWX('--libs')


    add_base_flags(module)
    module.include_dirs.extend(
        [x[2:] for x in wxFlags if x.startswith('-I')])


    module.define_macros.extend(
        [(x[2:], None) for x in wxFlags if x.startswith('-D')])
    module.undef_macros.extend(
        [x[2:] for x in wxFlags if x.startswith('-U')])

    module.libraries.extend(
        [x[2:] for x in wxLibs if x.startswith('-l')])
    module.library_dirs.extend(
        [x[2:] for x in wxLibs if x.startswith('-L')])
    module.extra_link_args.extend(
        [x for x in wxLibs if not (x.startswith('-l') or x.startswith('-L'))])


# Make sure you use the Tk version given by Tkinter.TkVersion
# or else you'll build for a wrong version of the Tcl
# interpreter (leading to nasty segfaults).

class FoundTclTk:
    pass

def find_tcltk():
    """Finds Tcl/Tk includes/libraries/version by interrogating Tkinter."""
    try:
        import Tkinter
    except:
        print "Tkinter not properly installed\n"
        sys.exit(1)
    if Tkinter.TkVersion < 8.3:
        print "Tcl/Tk v8.3 or later required\n"
        sys.exit(1)
    o = FoundTclTk()
    try:
        tk=Tkinter.Tk()
    except Tkinter.TclError:
        print "Using default library and include directories for Tcl and Tk because a"
        print "Tk window failed to open.  You may need to define DISPLAY for Tk to work"
        print "so that setup can determine where your libraries are located."
        o.tcl_lib = "/usr/local/lib"
        o.tcl_inc = "/usr/local/include"
        o.tk_lib = "/usr/local/lib"
        o.tk_inc = "/usr/local/include"
        o.tkv = ""
    else:
        tk.withdraw()
        o.tcl_lib = os.path.normpath(os.path.join(str(tk.getvar('tcl_library')), '../'))
        o.tk_lib = os.path.normpath(os.path.join(str(tk.getvar('tk_library')), '../'))
        o.tkv = str(Tkinter.TkVersion)[:3]
        o.tcl_inc = os.path.normpath(os.path.join(str(tk.getvar('tcl_library')),
                    '../../include/tcl'+o.tkv))
        if not os.path.exists(o.tcl_inc):
            o.tcl_inc = os.path.normpath(os.path.join(str(tk.getvar('tcl_library')),
                        '../../include'))
        o.tk_inc = os.path.normpath(os.path.join(str(tk.getvar('tk_library')),
                    '../../include/tk'+o.tkv))
        if not os.path.exists(o.tk_inc):
            o.tk_inc = os.path.normpath(os.path.join(str(tk.getvar('tk_library')),
                        '../../include'))

        if ((not os.path.exists(os.path.join(o.tk_inc,'tk.h'))) and
            os.path.exists(os.path.join(o.tcl_inc,'tk.h'))):
            o.tk_inc = o.tcl_inc

        if not os.path.exists(o.tcl_inc):
            # this is a hack for suse linux, which is broken
            if (sys.platform.startswith('linux') and
                os.path.exists('/usr/include/tcl.h') and
                os.path.exists('/usr/include/tk.h')):
                o.tcl_inc = '/usr/include/'
                o.tk_inc = '/usr/include/'

        if not os.path.exists(o.tcl_inc):
            print 'cannot find tcl/tk headers. giving up.'
            sys.exit()
    return o


def add_tk_flags(module):
    'Add the module flags to build extensions which use tk'
    if sys.platform=='win32':
        major, minor1, minor2, s, tmp = sys.version_info
        print 'building tkagg', major, minor1
        if major==2 and minor1 in [3, 4, 5]:
            print '\tBuilding for python2%d'%minor1
            module.include_dirs.extend(['win32_static/include/tcl84'])
            module.libraries.extend(['tk84', 'tcl84'])
        elif major==2 and minor1==2:
            print '\tBuilding for python22'
            module.include_dirs.extend(['win32_static/include/tcl83'])
            module.libraries.extend(['tk83', 'tcl83'])
        else:
            raise RuntimeError('No tk/win32 support for this python version yet')
        module.library_dirs.extend([os.path.join(sys.prefix, 'dlls')])
        return

    elif sys.platform == 'darwin' :
        # this config section lifted directly from Imaging - thanks to
        # the effbot!

        # First test for a MacOSX/darwin framework install
        from os.path import join, exists
        framework_dirs = [
            '/System/Library/Frameworks/',
            '/Library/Frameworks',
            join(os.getenv('HOME'), '/Library/Frameworks')
        ]

        # Find the directory that contains the Tcl.framwork and Tk.framework
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
                for fw in 'Tcl', 'Tk'
                for H in 'Headers', 'Versions/Current/PrivateHeaders'
            ]

            # For 8.4a2, the X11 headers are not included. Rather than include a
            # complicated search, this is a hard-coded path. It could bail out
            # if X11 libs are not found...
            # tk_include_dirs.append('/usr/X11R6/include')
            frameworks = ['-framework', 'Tcl', '-framework', 'Tk']
            module.include_dirs.extend(tk_include_dirs)
            module.extra_link_args.extend(frameworks)
            module.extra_compile_args.extend(frameworks)
            return

    # you're still here? ok we'll try it this way
    o = find_tcltk() # todo: try/except
    module.include_dirs.extend([o.tcl_inc, o.tk_inc])
    module.library_dirs.extend([o.tcl_lib, o.tk_lib])
    module.libraries.extend(['tk'+o.tkv, 'tcl'+o.tkv])


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
                       ['src/_windowing.cpp',
                        ],
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

    module = Extension('matplotlib.ft2font', deps)
    add_ft2font_flags(module)
    ext_modules.append(module)
    BUILT_FT2FONT = True

def build_gtkagg(ext_modules, packages, numerix):
    global BUILT_GTKAGG
    if BUILT_GTKAGG: return # only build it if you you haven't already
    deps = ['src/_gtkagg.cpp', 'src/mplutils.cpp']#, 'src/_transforms.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension('matplotlib.backends._gtkagg',
                       deps,
                       )


    # add agg flags before pygtk because agg only supports freetype1
    # and pygtk includes freetype2.  This is a bit fragile.


    add_agg_flags(module)
    add_ft2font_flags(module)
    add_pygtk_flags(module)

    ext_modules.append(module)
    BUILT_GTKAGG = True

def build_tkagg(ext_modules, packages, numerix):
    # XXX doesn't need numerix arg
    global BUILT_TKAGG
    if BUILT_TKAGG: return # only build it if you you haven't already
    deps = ['src/_tkagg.cpp']
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension('matplotlib.backends._tkagg',
                       deps,
                       )

    # add agg flags before pygtk because agg only supports freetype1
    # and pygtk includes freetype2.  This is a bit fragile.

    add_tk_flags(module) # do this first
    add_agg_flags(module)
    add_ft2font_flags(module)

    ext_modules.append(module)
    BUILT_TKAGG = True


def build_wxagg(ext_modules, packages, numerix, abortOnFailure):
     global BUILT_WXAGG
     if BUILT_WXAGG:
         return

     wxconfig = find_wx_config()

     # Avoid aborting the whole build process if `wx-config' can't be found and
     # BUILD_WXAGG in setup.py is set to "auto"
     if sys.platform == 'win32':
         # mingw32 cannot link against distributed wx libs
         # since they are built with VisualStudio
         if win32_compiler == 'mingw32':
             return
         else:
             pass

     elif wxconfig is None:
         print """
WXAgg's accelerator requires `wx-config'.

The `wx-config\' executable could not be located in any directory of the
PATH environment variable. If you want to build WXAgg, and wx-config is
in some other location or has some other name, set the WX_CONFIG
environment variable to the full path of the executable like so:

export WX_CONFIG=/usr/lib/wxPython-2.6.1.0-gtk2-unicode/bin/wx-config
"""
         if not abortOnFailure:
             print """Building MPL without wxAgg"""
             BUILT_WXAGG = True
             return
         else:
             sys.exit(1)
     elif not check_wxpython_headers(wxconfig):
         print 'WXAgg\'s accelerator requires the wxPython headers.'

         if not abortOnFailure:
             BUILT_WXAGG = True
             return
         else:
             print """
The wxPython header files could not be located in any of the standard
include
directories or include directories reported by `wx-config --cppflags'."""
             sys.exit(1)
     elif not check_wxpython_broken_macosx104_version(wxconfig):
         print 'WXAgg\'s accelerator not building because a broken wxPython (installed by Apple\'s Mac OS X) was found.'
         if not abortOnFailure:
             BUILT_WXAGG = True
             return
         else:
             print """
The wxPython installation is the broken version installed with Mac OS X 10.4."""
             sys.exit(1)

     deps = ['src/_wxagg.cpp', 'src/mplutils.cpp']
     deps.extend(glob.glob('CXX/*.cxx'))
     deps.extend(glob.glob('CXX/*.c'))

     module = Extension('matplotlib.backends._wxagg', deps)

     add_agg_flags(module)
     add_ft2font_flags(module)
     add_wx_flags(module, wxconfig)

     ext_modules.append(module)
     BUILT_WXAGG = True




def build_agg(ext_modules, packages, numerix):
    global BUILT_AGG
    if BUILT_AGG: return # only build it if you you haven't already


    agg = (
           'agg_trans_affine.cpp',
           'agg_path_storage.cpp',
           'agg_bezier_arc.cpp',
           'agg_curves.cpp',
           'agg_vcgen_dash.cpp',
           'agg_vcgen_stroke.cpp',
           #'agg_vcgen_markers_term.cpp',
           'agg_rasterizer_scanline_aa.cpp',
           'agg_image_filters.cpp',
           )

    if 'numarray' in numerix: # Build for numarray
        deps = ['%s/src/%s'%(AGG_VERSION,name) for name in agg]
        deps.extend(('src/_image.cpp', 'src/ft2font.cpp', 'src/mplutils.cpp'))
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))
        temp_copy('src/_backend_agg.cpp', 'src/_na_backend_agg.cpp')
        deps.append('src/_na_backend_agg.cpp')
        module = Extension(
            'matplotlib.backends._na_backend_agg',
            deps,
            include_dirs=numarray_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMARRAY=1')
        add_agg_flags(module)
        add_ft2font_flags(module)
        ext_modules.append(module)
    if 'Numeric' in numerix: # Build for Numeric
        deps = ['%s/src/%s'%(AGG_VERSION, name) for name in agg]
        deps.extend(('src/_image.cpp', 'src/ft2font.cpp', 'src/mplutils.cpp'))
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))

        temp_copy('src/_backend_agg.cpp', 'src/_nc_backend_agg.cpp')
        deps.append('src/_nc_backend_agg.cpp')
        module = Extension(
            'matplotlib.backends._nc_backend_agg',
            deps,
            include_dirs=numeric_inc_dirs,
            )

        module.extra_compile_args.append('-DNUMERIC=1')
        add_agg_flags(module)
        add_ft2font_flags(module)
        ext_modules.append(module)
    if 'numpy' in numerix: # Build for numpy
        deps = ['%s/src/%s'%(AGG_VERSION, name) for name in agg]
        deps.extend(('src/_image.cpp', 'src/ft2font.cpp', 'src/mplutils.cpp'))
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))

        temp_copy('src/_backend_agg.cpp', 'src/_ns_backend_agg.cpp')
        deps.append('src/_ns_backend_agg.cpp')
        module = Extension(
            'matplotlib.backends._ns_backend_agg',
            deps,
            include_dirs=numeric_inc_dirs,
            )

        add_numpy_flags(module)
        module.extra_compile_args.append('-DSCIPY=1')

        add_agg_flags(module)
        add_ft2font_flags(module)
        ext_modules.append(module)

    BUILT_AGG = True

def build_image(ext_modules, packages, numerix):
    global BUILT_IMAGE
    if BUILT_IMAGE: return # only build it if you you haven't already

    agg = ('agg_trans_affine.cpp',
           'agg_path_storage.cpp',
           'agg_rasterizer_scanline_aa.cpp',
           'agg_image_filters.cpp',
           'agg_bezier_arc.cpp',
           )

    if 'numarray' in numerix: # Build for numarray
        temp_copy('src/_image.cpp', 'src/_na_image.cpp')
        deps = ['src/_na_image.cpp', 'src/mplutils.cpp']
        deps.extend(['%s/src/%s'%(AGG_VERSION, name) for name in agg])
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))
        module = Extension(
            'matplotlib._na_image',
            deps,
            include_dirs=numarray_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMARRAY=1')
        add_agg_flags(module)
        ext_modules.append(module)

    if 'Numeric' in numerix: # Build for Numeric
        temp_copy('src/_image.cpp', 'src/_nc_image.cpp')
        deps = ['src/_nc_image.cpp', 'src/mplutils.cpp']
        deps.extend(['%s/src/%s'%(AGG_VERSION,name) for name in agg])
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))

        module = Extension(
            'matplotlib._nc_image',
            deps,
            include_dirs=numeric_inc_dirs,
            )

        module.extra_compile_args.append('-DNUMERIC=1')
        add_agg_flags(module)
        ext_modules.append(module)

    if 'numpy' in numerix: # Build for numpy
        temp_copy('src/_image.cpp', 'src/_ns_image.cpp')
        deps = ['src/_ns_image.cpp', 'src/mplutils.cpp']
        deps.extend(['%s/src/%s'%(AGG_VERSION,name) for name in agg])
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))

        module = Extension(
            'matplotlib._ns_image',
            deps,
            include_dirs=numeric_inc_dirs,
            )

        add_numpy_flags(module)
        module.extra_compile_args.append('-DSCIPY=1')
        add_agg_flags(module)
        ext_modules.append(module)

    BUILT_IMAGE = True

def build_swigagg(ext_modules, packages):
    # setup the swig agg wrapper
    deps = ['src/agg.cxx']
    deps.extend(['%s/src/%s'%(AGG_VERSION, fname) for fname in
                 (
        'agg_trans_affine.cpp',
        'agg_path_storage.cpp',
        'agg_bezier_arc.cpp',
        'agg_vcgen_dash.cpp',
        'agg_vcgen_stroke.cpp',
        'agg_rasterizer_scanline_aa.cpp',
        'agg_curves.cpp',
        )
                 ])


    agg = Extension('matplotlib._agg',
                    deps,
                    )

    agg.include_dirs.extend(['%s/include'%AGG_VERSION, 'src', 'swig'])
    agg.libraries.extend(std_libs)
    ext_modules.append(agg)

def build_transforms(ext_modules, packages, numerix):
    if 'numarray' in numerix: # Build for numarray
        cxx = glob.glob('CXX/*.cxx')
        cxx.extend(glob.glob('CXX/*.c'))
        temp_copy("src/_transforms.cpp","src/_na_transforms.cpp")
        module = Extension('matplotlib._na_transforms',
                             ['src/_na_transforms.cpp',
                              'src/mplutils.cpp'] + cxx,
                             libraries = std_libs,
                             include_dirs = ['src', '.']+numarray_inc_dirs,
                             )

        module.extra_compile_args.append("-DNUMARRAY=1")
        add_base_flags(module)
        ext_modules.append(module)

    if 'Numeric' in numerix:  # Build for Numeric
        cxx = glob.glob('CXX/*.cxx')
        cxx.extend(glob.glob('CXX/*.c'))
        temp_copy("src/_transforms.cpp","src/_nc_transforms.cpp")
        module = Extension('matplotlib._nc_transforms',
                             ['src/_nc_transforms.cpp',
                              'src/mplutils.cpp'] + cxx,
                             libraries = std_libs,
                             include_dirs = ['src', '.']+numeric_inc_dirs,
                             )



        module.extra_compile_args.append("-DNUMERIC=1")
        add_base_flags(module)
        ext_modules.append(module)

    if 'numpy' in numerix:  # Build for numpy
        cxx = glob.glob('CXX/*.cxx')
        cxx.extend(glob.glob('CXX/*.c'))
        temp_copy("src/_transforms.cpp","src/_ns_transforms.cpp")
        module = Extension('matplotlib._ns_transforms',
                             ['src/_ns_transforms.cpp',
                              'src/mplutils.cpp'] + cxx,
                             libraries = std_libs,
                             include_dirs = ['src', '.']+numeric_inc_dirs,
                             )


        add_numpy_flags(module)
        module.extra_compile_args.append("-DSCIPY=1")
        add_base_flags(module)
        ext_modules.append(module)

def build_enthought(ext_modules, packages):
    global BUILT_ENTHOUGHT
    if BUILT_ENTHOUGHT: return # only build it if you you haven't already

    ctraits = Extension('matplotlib.enthought.traits.ctraits',  ['lib/matplotlib/enthought/traits/ctraits.c'])
    ext_modules.append(ctraits)
    packages.extend(['matplotlib/enthought',
                     'matplotlib/enthought/traits',
                     'matplotlib/enthought/traits/ui',
                     'matplotlib/enthought/traits/ui/null',
                     'matplotlib/enthought/resource',
                     'matplotlib/enthought/util',
                     ])
    BUILT_ENTHOUGHT = True

def build_contour(ext_modules, packages, numerix):
    global BUILT_CONTOUR
    if BUILT_CONTOUR: return # only build it if you you haven't already

    if 'numarray' in numerix: # Build for numarray
        temp_copy('src/cntr.c', 'src/_na_cntr.c')
        module = Extension(
            'matplotlib._na_cntr',
            [  'src/_na_cntr.c',],
            include_dirs=numarray_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMARRAY=1')
        add_base_flags(module)
        ext_modules.append(module)

    if 'Numeric' in numerix: # Build for Numeric
        temp_copy('src/cntr.c', 'src/_nc_cntr.c')
        module = Extension(
            'matplotlib._nc_cntr',
            [ 'src/_nc_cntr.c'],
            include_dirs=numeric_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMERIC=1')
        add_base_flags(module)
        ext_modules.append(module)
    if 'numpy' in numerix: # Build for numpy
        temp_copy('src/cntr.c', 'src/_ns_cntr.c')
        module = Extension(
            'matplotlib._ns_cntr',
            [ 'src/_ns_cntr.c'],
            include_dirs=numeric_inc_dirs,
            )
        add_numpy_flags(module)
        module.extra_compile_args.append('-DSCIPY=1')
        add_base_flags(module)
        ext_modules.append(module)


    BUILT_CONTOUR = True


def build_nxutils(ext_modules, packages, numerix):
    global BUILT_NXUTILS
    if BUILT_NXUTILS: return # only build it if you you haven't already

    if 'numarray' in numerix: # Build for numarray
        temp_copy('src/nxutils.c', 'src/_na_nxutils.c')
        module = Extension(
            'matplotlib._na_nxutils',
            [  'src/_na_nxutils.c',],
            include_dirs=numarray_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMARRAY=1')
        add_base_flags(module)
        ext_modules.append(module)

    if 'Numeric' in numerix: # Build for Numeric
        temp_copy('src/nxutils.c', 'src/_nc_nxutils.c')
        module = Extension(
            'matplotlib._nc_nxutils',
            [ 'src/_nc_nxutils.c'],
            include_dirs=numeric_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMERIC=1')
        add_base_flags(module)
        ext_modules.append(module)
    if 'numpy' in numerix: # Build for numpy
        temp_copy('src/nxutils.c', 'src/_ns_nxutils.c')
        module = Extension(
            'matplotlib._ns_nxutils',
            [ 'src/_ns_nxutils.c'],
            include_dirs=numeric_inc_dirs,
            )
        add_numpy_flags(module)
        module.extra_compile_args.append('-DSCIPY=1')
        add_base_flags(module)
        ext_modules.append(module)


    BUILT_NXUTILS = True


def build_gdk(ext_modules, packages, numerix):
    global BUILT_GDK
    if BUILT_GDK: return # only build it if you you haven't already

    if 'numarray' in numerix: # Build for numarray
        temp_copy('src/_backend_gdk.c', 'src/_na_backend_gdk.c')
        module = Extension(
            'matplotlib.backends._na_backend_gdk',
            ['src/_na_backend_gdk.c', ],
            libraries = [],
            include_dirs=numarray_inc_dirs,
            )
        module.extra_compile_args.append('-DNUMARRAY=1')
        add_base_flags(module)
        add_pygtk_flags(module)
        ext_modules.append(module)

    if 'Numeric' in numerix:# Build for Numeric
        temp_copy('src/_backend_gdk.c', 'src/_nc_backend_gdk.c')
        module = Extension(
            'matplotlib.backends._nc_backend_gdk',
            ['src/_nc_backend_gdk.c', ],
            libraries = [],
            include_dirs=numeric_inc_dirs,
            )

        module.extra_compile_args.append('-DNUMERIC=1')
        add_base_flags(module)
        add_pygtk_flags(module)
        ext_modules.append(module)

    if 'numpy' in numerix:# Build for numpy
        temp_copy('src/_backend_gdk.c', 'src/_ns_backend_gdk.c')
        module = Extension(
            'matplotlib.backends._ns_backend_gdk',
            ['src/_ns_backend_gdk.c', ],
            libraries = [],
            include_dirs=numeric_inc_dirs,
            )

        add_numpy_flags(module)
        module.extra_compile_args.append('-DSCIPY=1')
        add_base_flags(module)
        add_pygtk_flags(module)
        ext_modules.append(module)

    BUILT_GDK = True

def build_subprocess(ext_modules, packages):
    module = Extension(
        'subprocess._subprocess',
        ['src/_subprocess.c', ],
        )
    add_base_flags(module)
    ext_modules.append(module)

def build_isnan(ext_modules, packages):
    module = Extension(
        'matplotlib._isnan',
        [ 'src/_isnan.c'],
        )
    add_base_flags(module)
    ext_modules.append(module)
