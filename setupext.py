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

WIN32

  If you are sufficiently masochistic that you want to build this
  yourself, download the win32_static dir from
  http://matplotlib.sourceforge.net/win32_static.tar.gz and
  see the README file in that dir

  > python setup.py build --compiler=mingw32 bdist_wininst  > build23.out

"""

import os

basedir = {
    'win32'  : ['win32_static',],
    'linux2' : ['/usr/local', '/usr',],
    'linux'  : ['/usr/local', '/usr',],
    'darwin' : ['/usr/local', '/usr', '/sw', '/usr/X11R6'],
    'freebsd4' : ['/usr/local', '/usr'],
    'sunos5' : [os.getenv('MPLIB_BASE') or '/usr/local',],
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
BUILT_WINDOWING = False
BUILT_CONTOUR   = False

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

def add_base_flags(module):
    incdirs = [os.path.join(p, 'include') for p in basedir[sys.platform]
               if os.path.exists(p)]
    libdirs = [os.path.join(p, 'lib')     for p in basedir[sys.platform]
               if os.path.exists(p)]
    module.include_dirs.extend(incdirs)
    module.library_dirs.extend(libdirs)


def getoutput(s):
    'get the output of a system command'

    ret =  os.popen(s).read().strip()
    return ret



def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    # before adding the freetype flags since -z comes later
    module.libraries.append('png')
    module.libraries.append('z')
    add_base_flags(module)
    module.include_dirs.extend(['src','agg22/include', '.'])

    # put these later for correct link order
    module.libraries.extend(['stdc++', 'm'])

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
            
    if sys.platform == 'win32':
        module.libraries.append('gw32c')

    # put this last for library link order     
    module.libraries.extend(['stdc++', 'm'])

    

def add_pygtk_flags(module):
    'Add the module flags to build extensions which use gtk'

    if sys.platform=='win32':
        # popen broken on my win32 plaform so I can't use pkgconfig
        module.library_dirs.extend(
            ['C:/GTK/bin', 'C:/GTK/lib'])

        module.include_dirs.extend(
            ['win32_static/include/pygtk-2.0',
             'C:/GTK/include',
             ])

    add_base_flags(module)

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
	o.tcl_lib = os.path.join((tk.getvar('tcl_library')), '../')
	o.tk_lib = os.path.join((tk.getvar('tk_library')), '../')
	o.tkv = str(Tkinter.TkVersion)[:3]
	o.tcl_inc = os.path.join((tk.getvar('tcl_library')), 
				 '../../include/tcl'+o.tkv)        
        if not os.path.exists(o.tcl_inc):
	    o.tcl_inc = os.path.join((tk.getvar('tcl_library')), 
				     '../../include')
	o.tk_inc = os.path.join((tk.getvar('tk_library')), 
				 '../../include/tk'+o.tkv)        
        if not os.path.exists(o.tk_inc):
	    o.tk_inc = os.path.join((tk.getvar('tk_library')), 
				     '../../include')
        if not os.path.exists(o.tcl_inc):
            print 'cannot find tcl/tk headers. giving up.'
            sys.exit()
    return o
	

def add_tk_flags(module):
    'Add the module flags to build extensions which use tk'
    if sys.platform=='win32':
        major, minor1, minor2, s, tmp = sys.version_info
        print 'building tkagg', major, minor1
        if major==2 and minor1==3:
            print '\tBuilding for python23'
            module.include_dirs.extend(['win32_static/include/tcl84'])
            module.library_dirs.extend(['C:/Python23/dlls'])        
            module.libraries.extend(['tk84', 'tcl84'])
        elif major==2 and minor1==2:
            print '\tBuilding for python22'
            module.include_dirs.extend(['win32_static/include/tcl83'])
            module.library_dirs.extend(['C:/Python22/dlls'])        
            module.libraries.extend(['tk83', 'tcl83'])
        else:
            raise RuntimeError('No tk/win32 support for this python version yet')
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
    o = find_tcltk()
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

def build_gtkagg(ext_modules, packages):
    global BUILT_GTKAGG
    if BUILT_GTKAGG: return # only build it if you you haven't already
    deps = ['src/_gtkagg.cpp', 'src/mplutils.cpp']
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

def build_tkagg(ext_modules, packages):
    global BUILT_TKAGG
    if BUILT_TKAGG: return # only build it if you you haven't already
    module = Extension('matplotlib.backends._tkagg',
                       ['src/_tkagg.cpp'],
                       )

    # add agg flags before pygtk because agg only supports freetype1
    # and pygtk includes freetype2.  This is a bit fragile.

    add_tk_flags(module) # do this first
    add_agg_flags(module)
    add_ft2font_flags(module)    
    ext_modules.append(module)    
    BUILT_TKAGG = True


def build_agg(ext_modules, packages):
    global BUILT_AGG
    if BUILT_AGG: return # only build it if you you haven't already
    
    deps = ['src/_backend_agg.cpp', 'src/ft2font.cpp', 'src/mplutils.cpp']
    deps.extend(glob.glob('agg22/src/*.cpp'))
    deps.extend(glob.glob('CXX/*.cxx'))
    deps.extend(glob.glob('CXX/*.c'))

    module = Extension(
        'matplotlib.backends._backend_agg',
        deps
        ,
        )

    add_agg_flags(module)
    add_ft2font_flags(module)
    ext_modules.append(module)    
    BUILT_AGG = True

def build_image(ext_modules, packages, numerix):
    global BUILT_IMAGE
    if BUILT_IMAGE: return # only build it if you you haven't already

    if numerix in ["numarray","both"]: # Build for numarray
        temp_copy('src/_image.cpp', 'src/_na_image.cpp')
        deps = ['src/_na_image.cpp', 'src/mplutils.cpp'] 
        deps.extend(glob.glob('agg22/src/*.cpp'))
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))
        module = Extension(
            'matplotlib._na_image',
            deps
            ,
            )    
        module.extra_compile_args.append('-DNUMARRAY=1')
        add_agg_flags(module)
        ext_modules.append(module)    

    if numerix in ["Numeric","both"]: # Build for Numeric
        temp_copy('src/_image.cpp', 'src/_nc_image.cpp')
        deps = ['src/_nc_image.cpp', 'src/mplutils.cpp'] 
        deps.extend(glob.glob('agg22/src/*.cpp'))
        deps.extend(glob.glob('CXX/*.cxx'))
        deps.extend(glob.glob('CXX/*.c'))
        module = Extension(
            'matplotlib._nc_image',
            deps
            ,
            )
        module.extra_compile_args.append('-DNUMERIC=1')
        add_agg_flags(module)
        ext_modules.append(module)

    BUILT_IMAGE = True

def build_transforms(ext_modules, packages, numerix):
    if numerix in ["numarray","both"]: # Build for numarray
        cxx = glob.glob('CXX/*.cxx')
        cxx.extend(glob.glob('CXX/*.c'))
        temp_copy("src/_transforms.cpp","src/_na_transforms.cpp")
        module = Extension('matplotlib._na_transforms',
                             ['src/_na_transforms.cpp',
                              'src/mplutils.cpp'] + cxx,
                             libraries = ['stdc++', 'm'],
                             include_dirs = ['src', '.'],
                             )
        module.extra_compile_args.append("-DNUMARRAY=1")
        ext_modules.append(module)
        
    if numerix in ["Numeric","both"]:  # Build for Numeric
        cxx = glob.glob('CXX/*.cxx')
        cxx.extend(glob.glob('CXX/*.c'))
        temp_copy("src/_transforms.cpp","src/_nc_transforms.cpp")
        module = Extension('matplotlib._nc_transforms',
                             ['src/_nc_transforms.cpp',
                              'src/mplutils.cpp'] + cxx,
                             libraries = ['stdc++', 'm'],
                             include_dirs = ['src', '.'],
                             )
        module.extra_compile_args.append("-DNUMERIC=1")
        ext_modules.append(module)
    

def build_contour(ext_modules, packages, numerix):
    global BUILT_CONTOUR
    if BUILT_CONTOUR: return # only build it if you you haven't already

    if numerix in ["numarray","both"]: # Build for numarray
        temp_copy('src/_contour.c', 'src/_na_contour.c')
        deps = ['src/_na_contour.c', 'src/gcntr.c'] 
        module = Extension(
            'matplotlib._na_contour',
            deps
            ,
            )    
        module.extra_compile_args.append('-DNUMARRAY=1')
        ext_modules.append(module)    

    if numerix in ["Numeric","both"]: # Build for Numeric
        temp_copy('src/_contour.c', 'src/_nc_contour.c')
        deps = ['src/_nc_contour.c', 'src/gcntr.c'] 
        module = Extension(
            'matplotlib._nc_contour',
            deps
            ,
            )
        module.extra_compile_args.append('-DNUMERIC=1')
        ext_modules.append(module)

    BUILT_CONTOUR = True
