"""
Some helper functions for building the C extensions

BUILDING ON WIN32

  * You need to make the cygwin import library.  Assuming you have a
    typical cygwin and python install, run the script importlib22.bat or
    importlib23.bat to build the python2.2 or 2.3 import libs

  * You need to following libraries (saved in win32_static):

      http://gnuwin32.sourceforge.net/downlinks/freetype.php
      http://gnuwin32.sourceforge.net/downlinks/zlib.php
      http://gnuwin32.sourceforge.net/downlinks/libgw32c.php
      http://gnuwin32.sourceforge.net/downlinks/libpng.php
      http://www.activestate.com/Products/Download/Download.plex?id=ActiveTcl

  * To install the gtk packages, you need pkg-config.  This is
    included in the GTK development lib.  You should have the GTK
    runtime and development libs installed to C:\GTK and make sure
    c:\GTK\lib and c:\GTK\bin are in your PATH.  Also, copy
    win32_static/pygtk-2.0.pc to c:\GTK\lib\pkgconfig
  
  * You must patch distutils for python23 or python22 to build agg
    with g++.  See
    http://mail.python.org/pipermail/distutils-sig/2004-January/003553.html.
    Edit c:/Python23/lib/distutils/cygwinccompiler.py and add the line
    to the two set_executables calls

      compiler_cxx='g++ -mcygwin -O -Wall',

  * build command

    > python setup.py build --compiler=mingw32 > build23.out

  * make the windows installer

    > python setup.py bdist_wininst

    Note on some systems this fails with a "extensions need to be
    built with the same version of the compiler" message.  The
    following workaround helps

    > python setup.py build --compiler=mingw32 bdist_wininst > build23.out
                        
    See for details http://groups.google.com/groups?hl=en&lr=&ie=UTF-8&oe=UTF-8&threadm=mailman.1060311735.32666.python-list%40python.org&rnum=1&prev=/groups%3Fhl%3Den%26lr%3D%26ie%3DUTF-8%26oe%3DUTF-8%26q%3Dpython%2B%2522extensions%2Bneed%2Bto%2Bbe%2Bbuilt%2Bwith%2Bthe%2Bsame%2Bversion%2Bof%2Bthe%2Bcompiler%2522 


    Current problems:

    c:\mingw\bin\dllwrap.exe -mno-cygwin -mdll -static --entry _DllMain@12 --output-lib build\temp.win32-2.2\Release\lib_tkagg.a --def build\temp.win32-2.2\Release\_tkagg.def -s build\temp.win32-2.2\Release\_tkagg.o -LC:/Python23/dlls -Lwin32_static -LC:\PYTHON22\libs -ltk84 -ltcl84 -lfreetype -lpng -lz -lgw32c -lstdc++ -lm -lpython22 -o build\lib.win32-2.2\matplotlib\backends\_tkagg.pyd
    error: command 'dllwrap' failed with exit status 1

    ideas: libstdc++ or libpython22 need to be moved up in link order?
    
"""
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
BUILT_FONTTOOLS = False
BUILT_GTKGD     = False
BUILT_GTKAGG    = False
BUILT_TKAGG     = False
BUILT_FT2FONT   = False

def getoutput(s):
    'get the output of a system command'

    ret =  os.popen(s).read().strip()
    return ret


def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    module.include_dirs.extend(['src','agg2/include'])
    module.libraries.extend(['freetype', 'png', 'z'])
    if sys.platform == 'win32':
        module.include_dirs.extend(
            ['win32_static/include',  ] )
        module.library_dirs.append('win32_static')
        module.libraries.append('gw32c')

    else:
        module.include_dirs.extend(
            ['/usr/include/freetype2',]
            )

    # put these later for correct link order
    module.libraries.extend(['stdc++', 'm'])


def add_gd_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.append('gd')


def add_ft2font_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.extend(['freetype', 'z'])
    if sys.platform == 'win32':
        module.include_dirs.extend(
            [  'win32_static/include',          # for ft2build.h
               'win32_static/include/freetype', # renamed from freetype2
               ]
            )
        module.library_dirs.append('win32_static')
        module.libraries.append('gw32c')
    else:
        module.include_dirs.extend(
            ['/usr/include', '/usr/include/freetype2',])

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
             'C:/GTK/include/glib-2.0',
             'C:/GTK/lib/glib-2.0/include',
             'C:/GTK/include/gtk-2.0',
             'C:/GTK/lib/gtk-2.0/include',
             'C:/GTK/include/atk-1.0',
             'C:/GTK/include/pango-1.0',
             ])
        
        module.libraries.extend([
            'gtk-win32-2.0', 'gdk-win32-2.0', 'atk-1.0',
            'gdk_pixbuf-2.0', 'pangowin32-1.0', 'gdi32',
            'pango-1.0', 'gobject-2.0', 'gmodule-2.0',
            'glib-2.0', 'intl', 'iconv'])

    else:
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


def add_tk_flags(module):
    'Add the module flags to build extensions which use tk'

    if sys.platform=='win32':
        module.include_dirs.extend(['win32_static/include/tcl'])
        module.library_dirs.extend(['C:/Python23/dlls'])
        module.libraries.extend(['tk84', 'tcl84'])
    else:
        module.libraries.extend(['tk', 'tcl'])


def build_ft2font(ext_modules, packages):
    global BUILT_FT2FONT
    if BUILT_FT2FONT: return # only build it if you you haven't already
    module = Extension('matplotlib.ft2font',
                       ['src/ft2font.c',
                        ],
                       )
    add_ft2font_flags(module)
    ext_modules.append(module)    
    BUILT_GTKGD = True

def build_gtkgd(ext_modules, packages):
    global BUILT_GTKGD
    if BUILT_GTKGD: return # only build it if you you haven't already
    module = Extension('matplotlib.backends._gtkgd',
                       ['src/_gtkgd.c'],
                       )
    add_pygtk_flags(module)
    add_gd_flags(module)
    ext_modules.append(module)    
    BUILT_GTKGD = True

def build_gtkagg(ext_modules, packages):
    global BUILT_GTKAGG
    if BUILT_GTKAGG: return # only build it if you you haven't already
    module = Extension('matplotlib.backends._gtkagg',
                       ['src/_gtkagg.cpp'],
                       )


    # add agg flags before pygtk because agg only supports freetype1
    # and pygtk includes freetype2.  This is a bit fragile.
    
    add_agg_flags(module)  
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
    ext_modules.append(module)    
    BUILT_AGG = True
    
def build_fonttools(ext_modules, packages):

    # don't build it if we have it
    #if sys.platform != 'win32': # always build for win32
    if 0:
        try: import ttfquery
        except ImportError: pass
        else: return
        
    global BUILT_FONTTOOLS

    if BUILT_FONTTOOLS: return # only build it if you you haven't already
    packages.extend(
        ['ttfquery',
         'FontTools',
         'FontTools.fontTools',
         'FontTools.fontTools.encodings',
         'FontTools.fontTools.misc',
         'FontTools.fontTools.ttLib',
         'FontTools.fontTools.ttLib.tables',
         'FontTools.fontTools.ttLib.test',
         ])

    ext_modules.append(
        Extension(
        'FontTools.fontTools.misc.eexecOp',
        ['FontToolsSrc/eexecOp/eexecOpmodule.c'],
        ))
    BUILT_FONTTOOLS = True





