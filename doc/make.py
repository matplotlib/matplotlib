#!/usr/bin/env python
"""Wrapper script for calling Sphinx. """

from __future__ import print_function
import glob
import os
import shutil
import sys
import re
import argparse
import subprocess
import matplotlib


def copy_if_out_of_date(original, derived):
    """Copy file only if newer as target or if target does not exist. """
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
    """Create target build directories if necessary. """
    build_dirs = ['build', 'build/doctrees', 'build/html', 'build/latex',
                  'build/texinfo', '_static', '_templates']
    for d in build_dirs:
        try:
            os.mkdir(d)
        except OSError:
            pass


def doctest():
    """Execute Sphinx 'doctest' target. """
    os.system('sphinx-build -b doctest -d build/doctrees . build/doctest')


def linkcheck():
    """Execute Sphinx 'linkcheck' target. """
    os.system('sphinx-build -b linkcheck -d build/doctrees . build/linkcheck')


# For generating PNGs of the top row of index.html:
FRONTPAGE_PY_PATH = "../examples/frontpage/"  # python scripts location
FRONTPAGE_PNG_PATH = "_static/"  # png files location
# png files and corresponding generation scripts:
FRONTPAGE_PNGS = {"surface3d_frontpage.png": "plot_3D.py",
                  "contour_frontpage.png":   "plot_contour.py",
                  "histogram_frontpage.png": "plot_histogram.py",
                  "membrane_frontpage.png":  "plot_membrane.py"}


def generate_frontpage_pngs(only_if_needed=True):
    """Executes the scripts for PNG generation of the top row of index.html.

    If `only_if_needed` is `True`, then the PNG file is only generated, if it
    doesn't exist or if the python file is newer.

    Note that the element `div.responsive_screenshots` in the file
    `_static/mpl.css` has the height and cumulative width of the used PNG files
    as attributes. This ensures that the magnification of those PNGs is <= 1.
    """
    for fn_png, fn_py in FRONTPAGE_PNGS.items():
        pn_png = os.path.join(FRONTPAGE_PNG_PATH, fn_png)  # get full paths
        pn_py = os.path.join(FRONTPAGE_PY_PATH, fn_py)

        # Read file modification times:
        mtime_py = os.path.getmtime(pn_py)
        mtime_png = (os.path.getmtime(pn_png) if os.path.exists(pn_png) else
                     mtime_py - 1)  # set older time, if file doesn't exist

        if only_if_needed and mtime_py <= mtime_png:
            continue  # do nothing if png is newer

        # Execute python as subprocess (preferred over os.system()):
        subprocess.check_call(["python", pn_py])  # raises CalledProcessError()
        os.rename(fn_png, pn_png)  # move file to _static/ directory


def html(buildername='html'):
    """Build Sphinx 'html' target. """
    check_build()
    generate_frontpage_pngs()

    rc = '../lib/matplotlib/mpl-data/matplotlibrc'
    default_rc = os.path.join(matplotlib._get_data_path(), 'matplotlibrc')
    if not os.path.exists(rc) and os.path.exists(default_rc):
        rc = default_rc
    copy_if_out_of_date(rc, '_static/matplotlibrc')

    if small_docs:
        options = "-D plot_formats=png:100"
    else:
        options = ''
    if warnings_as_errors:
        options = options + ' -W'
    if os.system('sphinx-build -j %d %s -b %s -d build/doctrees . build/%s' % (
            n_proc, options, buildername, buildername)):
        raise SystemExit("Building HTML failed.")

    # Clean out PDF files from the _images directory
    for filename in glob.glob('build/%s/_images/*.pdf' % buildername):
        os.remove(filename)

    shutil.copy('../CHANGELOG', 'build/%s/_static/CHANGELOG' % buildername)


def htmlhelp():
    """Build Sphinx 'htmlhelp' target. """
    html(buildername='htmlhelp')
    # remove scripts from index.html
    with open('build/htmlhelp/index.html', 'r+') as fh:
        content = fh.read()
        fh.seek(0)
        content = re.sub(r'<script>.*?</script>', '', content,
                         flags=re.MULTILINE | re.DOTALL)
        fh.write(content)
        fh.truncate()


def latex():
    """Build Sphinx 'latex' target. """
    check_build()
    # figs()
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
    """Build Sphinx 'texinfo' target. """
    check_build()
    # figs()
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
    """Remove generated files. """
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("examples", ignore_errors=True)
    for pattern in ['mpl_examples/api/*.png',
                    'mpl_examples/pylab_examples/*.png',
                    'mpl_examples/pylab_examples/*.pdf',
                    'mpl_examples/units/*.png',
                    'mpl_examples/pyplots/tex_demo.png',
                    '_static/matplotlibrc',
                    '_templates/gallery.html',
                    'users/installing.rst']:
        for filename in glob.glob(pattern):
            if os.path.exists(filename):
                os.remove(filename)
        for fn in FRONTPAGE_PNGS.keys():  # remove generated PNGs
            pn = os.path.join(FRONTPAGE_PNG_PATH, fn)
            if os.path.exists(pn):
                os.remove(os.path.join(pn))


def build_all():
    """Build Sphinx 'html' and 'latex' target. """
    # figs()
    html()
    latex()


funcd = {
    'html':      html,
    'htmlhelp':  htmlhelp,
    'latex':     latex,
    'texinfo':   texinfo,
    'clean':     clean,
    'all':       build_all,
    'doctest':   doctest,
    'linkcheck': linkcheck,
    }


small_docs = False
warnings_as_errors = True
n_proc = 1

# Change directory to the one containing this file
current_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.join(current_dir, __file__)))
copy_if_out_of_date('../INSTALL', 'users/installing.rst')

# Create the examples symlink, if it doesn't exist

required_symlinks = [
    ('mpl_examples', '../examples/'),
    ('mpl_toolkits/axes_grid1/examples', '../../../examples/axes_grid1/'),
    ('mpl_toolkits/axisartist/examples', '../../../examples/axisartist/')
    ]

symlink_warnings = []
for link, target in required_symlinks:
    if sys.platform == 'win32' and os.path.isfile(link):
        # This is special processing that applies on platforms that don't deal
        # with git symlinks -- probably only MS windows.
        delete = False
        with open(link, 'r') as link_content:
            delete = target == link_content.read()
        if delete:
            symlink_warnings.append('deleted:  doc/{0}'.format(link))
            os.unlink(link)
        else:
            raise RuntimeError("doc/{0} should be a directory or symlink -- it"
                               " isn't".format(link))
    if not os.path.exists(link):
        try:
            os.symlink(os.path.normcase(target), link)
        except OSError:
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
parser.add_argument("--allowsphinxwarnings",
                    help="Don't turn Sphinx warnings into errors",
                    action="store_true")
parser.add_argument("-n",
                    help="Number of parallel workers to use")

args = parser.parse_args()
if args.small:
    small_docs = True
if args.allowsphinxwarnings:
    warnings_as_errors = False
if args.n is not None:
    n_proc = int(args.n)

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
