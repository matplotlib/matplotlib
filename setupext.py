
"""
Some helper functions for building the C extensions
"""
import sys, os
from distutils.core import Extension

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


def build_gtkgd(ext_modules):
    module = Extension('matplotlib.backends._gtkgd',
                       ['src/_gtkgd.c'],
                       )
    add_pygtk_flags(module)
    add_gd_flags(module)
    ext_modules.append(module)    


def add_agg_flags(module):
    'Add the module flags to build extensions which use gtk'
    include_dirs = ['/usr/X11R6/include', '/home/jdhunter/c/src/agg2/include',
                    '/usr/include/freetype1', 'src']
    library_dirs = ['/usr/X11R6/lib', '/home/jdhunter/c/src/agg2/src']
    libraries = ['agg', 'X11', 'm', 'freetype']
    extra_link_args = []
    module.include_dirs.extend(include_dirs)
    module.libraries.extend(libraries)
    module.library_dirs.extend(library_dirs)
    module.extra_link_args.extend(extra_link_args)

def build_agg(ext_modules):
    module = Extension('matplotlib.backends._backend_agg',
                       ['src/_backend_agg.cpp'],
                       )
    add_agg_flags(module)
    ext_modules.append(module)    

