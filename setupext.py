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

  > python setup.py build --compiler=mingw32 bdist_wininst --install-script postinstall.py > build23.out

"""

import os

basedir = {

    'win32' : 'win32_static',
    'linux2' : '/usr',
    'linux'  : '/usr',
    'darwin' : '/usr/local',
    'sunos5' : os.getenv('MPLIB_BASE') or '/usr/local'
}

import sys, os
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
BUILT_IMAGE   = False
BUILT_TKAGG     = False

def getoutput(s):
    'get the output of a system command'

    ret =  os.popen(s).read().strip()
    return ret


def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    # before adding the freetype flags since -z comes later
    module.libraries.append('png')  
    module.include_dirs.append(os.path.join(basedir[sys.platform], 'include'))
    module.include_dirs.extend(['src','agg2/include'])
    module.library_dirs.append(os.path.join(basedir[sys.platform], 'lib'))



    # put these later for correct link order
    module.libraries.extend(['stdc++', 'm'])

def add_gd_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.append('gd')


def add_ft2font_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.extend(['freetype', 'z'])

    inc = os.path.join(basedir[sys.platform], 'include')
    module.include_dirs.append(inc)
    module.include_dirs.append(os.path.join(inc, 'freetype2'))

    module.library_dirs.append(os.path.join(basedir[sys.platform], 'lib'))

    if sys.platform == 'win32':
        module.libraries.append('gw32c')

    # put this last for library link order     
    module.libraries.append('m')

    

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
        [flag[2:] for dir in linkerFlags if flag.startswith('-L')])


    module.extra_link_args.extend(
        [flag for flag in linkerFlags if not
         (flag.startswith('-l') or flag.startswith('-L'))])


# Make sure you use the Tk version given by Tkinter.TkVersion
# or else you'll build for a wrong version of the Tcl
# interpreter (leading to nasty segfaults).

class found_tcltk:
    pass

def find_tcltk():
    """Finds Tcl/Tk includes/libraries/version by interrogating Tkinter."""
    try:
	import Tkinter
    except:
	print "Tkinter not properly installed\n"
	exit(1)
    if Tkinter.TkVersion < 8.3:
	print "Tcl/Tk v8.3 or later required\n"
	exit(1)
    tk=Tkinter.Tk()
    tk.withdraw()
    o = found_tcltk()
    o.tcl_lib = os.path.join((tk.getvar('tcl_library')), '../')
    o.tcl_inc = os.path.join((tk.getvar('tcl_library')), '../../include')
    o.tk_lib = os.path.join((tk.getvar('tk_library')), '../')
    o.tkv = str(Tkinter.TkVersion)[:3]
    return o

def add_tk_flags(module):
    'Add the module flags to build extensions which use tk'
    if sys.platform=='win32':
        module.include_dirs.extend(['win32_static/include/tcl'])
        module.library_dirs.extend(['C:/Python23/dlls'])
        module.libraries.extend(['tk84', 'tcl84'])
    elif sys.platform == 'darwin':
        module.extra_link_args.extend(['-framework','Tcl'])
        module.extra_link_args.extend(['-framework','Tk'])
    else:
	o = find_tcltk()
	module.include_dirs.append(o.tcl_inc)
	module.library_dirs.extend([o.tcl_lib, o.tk_lib])
        module.libraries.extend(['tk'+o.tkv, 'tcl'+o.tkv])

def build_ft2font(ext_modules, packages):
    global BUILT_FT2FONT
    if BUILT_FT2FONT: return # only build it if you you haven't already
    module = Extension('matplotlib.ft2font',
                       ['src/ft2font.c',
                        ],
                       )
    add_ft2font_flags(module)
    ext_modules.append(module)    
    BUILT_FT2FONT = True

def build_gtkagg(ext_modules, packages):
    global BUILT_GTKAGG
    if BUILT_GTKAGG: return # only build it if you you haven't already
    module = Extension('matplotlib.backends._gtkagg',
                       ['src/_gtkagg.cpp'],
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
    
    deps = ['src/_backend_agg.cpp', 'src/ft2font.c'] 
    deps.extend(glob.glob('agg2/src/*.cpp'))
                       
    module = Extension(
        'matplotlib.backends._backend_agg',
        deps
        ,
        )

    add_agg_flags(module)
    add_ft2font_flags(module)
    ext_modules.append(module)    
    BUILT_AGG = True

def build_image(ext_modules, packages):
    global BUILT_IMAGE
    if BUILT_IMAGE: return # only build it if you you haven't already
    
    deps = ['src/_image.cpp'] 
    deps.extend(glob.glob('agg2/src/*.cpp'))
                       
    module = Extension(
        'matplotlib._image',
        deps
        ,
        )
    add_agg_flags(module)
    ext_modules.append(module)    
    BUILT_IMAGE = True
    
