#!/usr/bin/env python

from __future__ import print_function
import glob
import os
import shutil
import sys
import re
import argparse

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

def html(buildername='html'):
    check_build()
    copy_if_out_of_date('../lib/matplotlib/mpl-data/matplotlibrc', '_static/matplotlibrc')
    if small_docs:
        options = "-D plot_formats=png:80"
    else:
        options = ''
    if warnings_as_errors:
        options = options + ' -W'
    if os.system('sphinx-build %s -b %s -d build/doctrees . build/%s' % (options, buildername, buildername)):
        raise SystemExit("Building HTML failed.")

    # Clean out PDF files from the _images directory
    for filename in glob.glob('build/%s/_images/*.pdf' % buildername):
        os.remove(filename)

    shutil.copy('../CHANGELOG', 'build/%s/_static/CHANGELOG' % buildername)

def htmlhelp():
    html(buildername='htmlhelp')
    # remove scripts from index.html
    with open('build/htmlhelp/index.html', 'r+') as fh:
        content = fh.read()
        fh.seek(0)
        content = re.sub(r'<script>.*?</script>', '', content, 
                         flags=re.MULTILINE| re.DOTALL)
        fh.write(content)
        fh.truncate()

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
    'htmlhelp' : htmlhelp,
    'latex'    : latex,
    'texinfo'  : texinfo,
    'clean'    : clean,
    'all'      : all,
    'doctest'  : doctest,
    'linkcheck': linkcheck,
    }


small_docs = False
warnings_as_errors = False

# Change directory to the one containing this file
current_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.join(current_dir, __file__)))
copy_if_out_of_date('../INSTALL', 'users/installing.rst')

# Create the examples symlink, if it doesn't exist

required_symlinks = [
    ('mpl_examples', '../examples/'),
    ('mpl_toolkits/axes_grid/examples', '../../../examples/axes_grid/')
    ]

symlink_warnings = []
for link, target in required_symlinks:
    if sys.platform == 'win32' and os.path.isfile(link):
        # This is special processing that applies on platforms that don't deal
        # with git symlinks -- probably only MS windows.
        delete = False
        with open(link, 'r') as content:
            delete = target == content.read()
        if delete:
            symlink_warnings.append('deleted:  doc/{0}'.format(link))
            os.unlink(link)
        else:
            raise RuntimeError("doc/{0} should be a directory or symlink -- it"
                               " isn't")
    if not os.path.exists(link):
        if hasattr(os, 'symlink'):
            os.symlink(target, link)
        else:
            symlink_warnings.append('files copied to {0}'.format(link))
            shutil.copytree(os.path.join(link, '..', target), link)

if sys.platform == 'win32' and len(symlink_warnings) > 0:
    print('The following items related to symlinks will show up '
          'as spurious changes in your \'git status\':\n\t{0}'
                    .format('\n\t'.join(symlink_warnings)))

parser = argparse.ArgumentParser(description='Build matplotlib docs')
parser.add_argument("cmd", help=("Command to execute. Can be multiple. "
                    "Valid options are: %s" % (funcd.keys())), nargs='*')
parser.add_argument("--small",
                    help="Smaller docs with only low res png figures",
                    action="store_true")
parser.add_argument("--warningsaserrors",
                    help="Turn Sphinx warnings into errors",
                    action="store_true")
args = parser.parse_args()
if args.small:
    small_docs = True
if args.warningsaserrors:
    warnings_as_errors = True

if args.cmd:
    for command in args.cmd:
        func = funcd.get(command)
        if func is None:
            raise SystemExit(('Do not know how to handle %s; valid commands'
                              ' are %s' % (command, funcd.keys())))
        func()
else:
    all()
os.chdir(current_dir)
