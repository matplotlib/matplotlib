"""
Some helper functions for building the C extensions
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
    return os.popen(s).read().strip()


def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    module.include_dirs.extend(['src','agg2/include'])
    module.libraries.extend(['freetype', 'png', 'z'])
    if sys.platform == 'win32':
        module.include_dirs.extend(
            ['c:/GnuWin32/include',  ] )
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
            [  'c:/GnuWin32/include', # for ft2build.h
               'c:/GnuWin32/include/freetype', # renamed from freetype2
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

    if sys.platform=='win32':
        module.library_dirs.extend(['C:\\GTK\\bin', 'C:\\GTK\\lib'])
    module.extra_link_args.extend(
        [flag for flag in linkerFlags if not
         (flag.startswith('-l') or flag.startswith('-L'))])



    print module.include_dirs
    print module.libraries
    print module.library_dirs
    

def add_tk_flags(module):
    'Add the module flags to build extensions which use tk'

    if sys.platform=='win32':
        module.include_dirs.extend(['C:\\Tcl\\include'])
        module.library_dirs.extend(['C:\\Tcl\\lib'])
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





