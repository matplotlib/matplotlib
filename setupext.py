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

def getoutput(s):
    'get the output of a system command'
    return os.popen(s).read().strip()


def add_pygtk_flags(module):
    'Add the module flags to build extensions which use gtk'
    pygtkIncludes = getoutput('pkg-config --cflags-only-I pygtk-2.0').split()
    gtkIncludes = getoutput('pkg-config --cflags-only-I gtk+-2.0').split()
    includes = pygtkIncludes + gtkIncludes
    include_dirs = [include[2:] for include in includes]


    pygtkLinker = getoutput('pkg-config --libs pygtk-2.0').split()
    gtkLinker =  getoutput('pkg-config --libs gtk+-2.0').split()
    linkerFlags = pygtkLinker + gtkLinker 

    libraries = [flag[2:] for flag in linkerFlags if flag.startswith('-l')]

    library_dirs = [flag[2:] for dir in linkerFlags if flag.startswith('-L')]

    extra_link_args = [flag for flag in linkerFlags if not
                       (flag.startswith('-l') or flag.startswith('-L'))]

    module.include_dirs.extend(include_dirs)
    module.libraries.extend(libraries)
    module.library_dirs.extend(library_dirs)
    module.extra_link_args.extend(extra_link_args)


def add_gd_flags(module):
    'Add the module flags to build extensions which use gd'
    module.libraries.append('gd')


def build_gtkgd(ext_modules, packages):
    global builtGTKGD
    if builtGTKGD: return 
    module = Extension('matplotlib.backends._gtkgd',
                       ['src/_gtkgd.c'],
                       )
    add_pygtk_flags(module)
    add_gd_flags(module)
    ext_modules.append(module)    
    builtGTKGD = True


def add_agg_flags(module):
    'Add the module flags to build extensions which use gtk'
    include_dirs = ['src','agg2/include']
    if sys.platform=='win32': include_dirs.extend(
        [  # build these libs on win32
        'win32src/freetype1',
        'win32src/libpng',
        'win32src/zlib',
        ]
        )
    else:
        include_dirs.extend(
            ['/usr/include/freetype1',]
            )

    library_dirs = []
    libraries = ['stdc++']
    if sys.platform != 'win32':
        libraries.extend(['z', 'ttf', 'png'])

    extra_link_args = []
    module.include_dirs.extend(include_dirs)
    module.libraries.extend(libraries)
    module.library_dirs.extend(library_dirs)
    module.extra_link_args.extend(extra_link_args)

def build_agg(ext_modules, packages):
    global builtAgg
    if builtAgg: return
    
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
    builtAgg = False
    
def build_fonttools(ext_modules, packages):

    global builtFonttools

    # only build them if not already installed
    try: import ttfquery
    except ImportError: pass
    else:
        builtFonttools = True
        return 

    if builtFonttools: return 
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





