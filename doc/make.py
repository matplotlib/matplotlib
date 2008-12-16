#!/usr/bin/env python
import fileinput
import glob
import os
import shutil
import sys

def check_build():
    build_dirs = ['build', 'build/doctrees', 'build/html', 'build/latex',
                  '_static', '_templates']
    for d in build_dirs:
        try:
            os.mkdir(d)
        except OSError:
            pass

def sf():
    'push a copy to the sf site'
    shutil.copy('../CHANGELOG', 'build/html/_static/CHANGELOG')
    os.system('cd build/html; rsync -avz . jdh2358,matplotlib@web.sf.net:/home/groups/m/ma/matplotlib/htdocs/ -essh --cvs-exclude')

def sfpdf():
    'push a copy to the sf site'
    os.system('cd build/latex; scp Matplotlib.pdf jdh2358,matplotlib@web.sf.net:/home/groups/m/ma/matplotlib/htdocs/')

def figs():
    os.system('cd users/figures/ && python make.py')

def examples():
    'make the rest examples'

    os.system('cd examples; svn-clean; python gen_rst.py')
    #pass

def gallery():
    'make the thumbnail gallery'
    os.system('cd _templates; python gen_gallery.py')


def html():
    check_build()
    if not os.path.exists('examples/index.rst'):
        examples()
    shutil.copy('../lib/matplotlib/mpl-data/matplotlibrc', '_static/matplotlibrc')
    #figs()
    if os.system('sphinx-build -b html -d build/doctrees . build/html'):
        raise SystemExit("Building HTML failed.")

    figures_dest_path = 'build/html/pyplots'
    if os.path.exists(figures_dest_path):
        shutil.rmtree(figures_dest_path)
    shutil.copytree('pyplots', figures_dest_path)

    # rebuild the gallery
    gallery()
    print 'Just rebuilt gallery, you may need to make html again'

def latex():
    check_build()
    #figs()
    if sys.platform != 'win32':
        # LaTeX format.
        if os.system('sphinx-build -b latex -d build/doctrees . build/latex'):
            raise SystemExit("Building LaTeX failed.")

        # Produce pdf.
        os.chdir('build/latex')

        # Copying the makefile produced by sphinx...
        if (os.system('pdflatex Matplotlib.tex') or
            os.system('pdflatex Matplotlib.tex') or
            os.system('makeindex -s python.ist Matplotlib.idx') or
            os.system('makeindex -s python.ist modMatplotlib.idx') or
            os.system('pdflatex Matplotlib.tex')):
            raise SystemExit("Rendering LaTeX failed.")

        os.chdir('../..')
    else:
        print 'latex build has not been tested on windows'

def clean():
    os.system('svn-clean')

def all():
    #figs()
    examples()
    html()
    latex()


funcd = {
    'figs'     : figs,
    'html'     : html,
    'latex'    : latex,
    'clean'    : clean,
    'sf'       : sf,
    'sfpdf'    : sfpdf,
    'examples' : examples,
    'gallery'  : gallery,
    'all'      : all,
    }


if len(sys.argv)>1:
    for arg in sys.argv[1:]:
        func = funcd.get(arg)
        if func is None:
            raise SystemExit('Do not know how to handle %s; valid args are %s'%(
                    arg, funcd.keys()))
        func()
else:
    all()
