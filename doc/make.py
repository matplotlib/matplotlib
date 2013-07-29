#!/usr/bin/env python

from __future__ import print_function
import fileinput
import glob
import os
import shutil
import sys

### Begin compatibility block for pre-v2.6: ###
#
# ignore_patterns and copytree funtions are copies of what is included
# in shutil.copytree of python v2.6 and later.
#
### When compatibility is no-longer needed, this block
### can be replaced with:
###
###     from shutil import ignore_patterns, copytree
###
### or the "shutil." qualifier can be prepended to the function
### names where they are used.

try:
    WindowsError
except NameError:
    WindowsError = None

def ignore_patterns(*patterns):
    """Function that can be used as copytree() ignore parameter.

    Patterns is a sequence of glob-style patterns
    that are used to exclude files"""
    import fnmatch
    def _ignore_patterns(path, names):
        ignored_names = []
        for pattern in patterns:
            ignored_names.extend(fnmatch.filter(names, pattern))
        return set(ignored_names)
    return _ignore_patterns

def copytree(src, dst, symlinks=False, ignore=None):
    """Recursively copy a directory tree using copy2().

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    XXX Consider this example code rather than the ultimate tool.

    """
    from shutil import copy2, Error, copystat
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy2(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.extend((src, dst, str(why)))
    if errors:
        raise Error(errors)

### End compatibility block for pre-v2.6 ###


def copy_if_out_of_date(original, derived):
    if (not os.path.exists(derived) or
        os.stat(derived).st_mtime < os.stat(original).st_mtime):
        try:
            shutil.copyfile(original, derived)
        except IOError:
            if os.path.basename(original) == 'matplotlibrc':
                msg = "'%s' not found. " % original + \
                      "Did you run `python setup.py build`?"
                raise IOError(msg)
            else:
                raise

def check_build():
    build_dirs = ['build', 'build/doctrees', 'build/html', 'build/latex',
                  'build/texinfo', '_static', '_templates']
    for d in build_dirs:
        try:
            os.mkdir(d)
        except OSError:
            pass

def doctest():
    os.system('sphinx-build -b doctest -d build/doctrees . build/doctest')

def linkcheck():
    os.system('sphinx-build -b linkcheck -d build/doctrees . build/linkcheck')

def html():
    check_build()
    copy_if_out_of_date('../lib/matplotlib/mpl-data/matplotlibrc', '_static/matplotlibrc')
    if small_docs:
        options = "-D plot_formats=\"[('png', 80)]\""
    else:
        options = ''
    if os.system('sphinx-build %s -b html -d build/doctrees . build/html' % options):
        raise SystemExit("Building HTML failed.")

    figures_dest_path = 'build/html/pyplots'
    if os.path.exists(figures_dest_path):
        shutil.rmtree(figures_dest_path)
    copytree(
        'pyplots', figures_dest_path,
        ignore=ignore_patterns("*.pyc"))

    # Clean out PDF files from the _images directory
    for filename in glob.glob('build/html/_images/*.pdf'):
        os.remove(filename)

    shutil.copy('../CHANGELOG', 'build/html/_static/CHANGELOG')

def latex():
    check_build()
    #figs()
    if sys.platform != 'win32':
        # LaTeX format.
        if os.system('sphinx-build -b latex -d build/doctrees . build/latex'):
            raise SystemExit("Building LaTeX failed.")

        # Produce pdf.
        os.chdir('build/latex')

        # Call the makefile produced by sphinx...
        if os.system('make'):
            raise SystemExit("Rendering LaTeX failed.")

        os.chdir('../..')
    else:
        print('latex build has not been tested on windows')

def texinfo():
    check_build()
    #figs()
    if sys.platform != 'win32':
        # Texinfo format.
        if os.system(
                'sphinx-build -b texinfo -d build/doctrees . build/texinfo'):
            raise SystemExit("Building Texinfo failed.")

        # Produce info file.
        os.chdir('build/texinfo')

        # Call the makefile produced by sphinx...
        if os.system('make'):
            raise SystemExit("Rendering Texinfo failed.")

        os.chdir('../..')
    else:
        print('texinfo build has not been tested on windows')

def clean():
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("examples", ignore_errors=True)
    for pattern in ['mpl_examples/api/*.png',
                    'mpl_examples/pylab_examples/*.png',
                    'mpl_examples/pylab_examples/*.pdf',
                    'mpl_examples/units/*.png',
                    'pyplots/tex_demo.png',
                    '_static/matplotlibrc',
                    '_templates/gallery.html',
                    'users/installing.rst']:
        for filename in glob.glob(pattern):
            if os.path.exists(filename):
                os.remove(filename)

def all():
    #figs()
    html()
    latex()


funcd = {
    'html'     : html,
    'latex'    : latex,
    'texinfo'  : texinfo,
    'clean'    : clean,
    'all'      : all,
    'doctest'  : doctest,
    'linkcheck': linkcheck,
    }


small_docs = False

# Change directory to the one containing this file
current_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.join(current_dir, __file__)))
copy_if_out_of_date('../INSTALL', 'users/installing.rst')

# Create the examples symlink, if it doesn't exist

required_symlinks = [
    ('mpl_examples', '../examples/'),
    ('mpl_toolkits/axes_grid/examples', '../../../examples/axes_grid/')
    ]

for link, target in required_symlinks:
    if not os.path.exists(link):
        if hasattr(os, 'symlink'):
            os.symlink(target, link)
        else:
            shutil.copytree(os.path.join(link, '..', target), link)

if len(sys.argv)>1:
    if '--small' in sys.argv[1:]:
        small_docs = True
        sys.argv.remove('--small')
    for arg in sys.argv[1:]:
        func = funcd.get(arg)
        if func is None:
            raise SystemExit('Do not know how to handle %s; valid args are %s'%(
                    arg, funcd.keys()))
        func()
else:
    small_docs = False
    all()
os.chdir(current_dir)
