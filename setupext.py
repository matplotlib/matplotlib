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

builtAgg       = False
builtFonttools = False
builtGTKGD     = False
builtGTKAgg    = False

def getoutput(s):
    'get the output of a system command'
    return os.popen(s).read().strip()


def add_agg_flags(module):
    'Add the module flags to build extensions which use agg'

    module.include_dirs.extend(['src','agg2/include'])
    if sys.platform=='win32':
        module.include_dirs.extend(
            [  # build these libs on win32
            'win32src/freetype1',
            'win32src/libpng',
            'win32src/zlib',
            ]
            )
    else:
        module.include_dirs.extend(
            ['/usr/include/freetype1',]
            )


    if sys.platform != 'win32':
        module.libraries.extend(['z', 'ttf', 'png'])


    module.libraries.extend(['stdc++'])

def add_gd_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.append('gd')


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

    module.extra_link_args.extend(
        [flag for flag in linkerFlags if not
         (flag.startswith('-l') or flag.startswith('-L'))])



def build_gtkgd(ext_modules, packages):
    global builtGTKGD
    if builtGTKGD: return # only build it if you you haven't already
    module = Extension('matplotlib.backends._gtkgd',
                       ['src/_gtkgd.c'],
                       )
    add_pygtk_flags(module)
    add_gd_flags(module)
    ext_modules.append(module)    
    builtGTKGD = True

def build_gtkagg(ext_modules, packages):
    global builtGTKAgg
    if builtGTKAgg: return # only build it if you you haven't already
    module = Extension('matplotlib.backends._gtkagg',
                       ['src/_gtkagg.cpp'],
                       )


    # add agg flags before pygtk because agg only supports freetype1
    # and pygtk includes freetype2.  This is a bit fragile.
    
    add_agg_flags(module)  
    add_pygtk_flags(module)

    ext_modules.append(module)    
    builtGTKAgg = True



def build_agg(ext_modules, packages):
    global builtAgg
    if builtAgg: return # only build it if you you haven't already
    
    deps = ['src/_backend_agg.cpp', 'src/font.cpp'] 
    deps.extend(glob.glob('agg2/src/*.cpp'))
    if sys.platform=='win32':
        deps.extend(glob.glob('win32src/freetype1/freetype/*.c'))
        deps.extend(glob.glob('win32src/libpng/*.c'))
        deps.extend(glob.glob('win32src/zlib/*.c'))
                       
    module = Extension(
        'matplotlib.backends._backend_agg',
        deps
        ,
        )
    add_agg_flags(module)
    ext_modules.append(module)    
    builtAgg = True
    
def build_fonttools(ext_modules, packages):

    global builtFonttools

    # only build them if not already installed
    try: import ttfquery
    except ImportError: pass
    else:
        builtFonttools = True
        return 

    if builtFonttools: return # only build it if you you haven't already
    packages.extend(
        ['ttfquery',
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
    builtFonttools = True





