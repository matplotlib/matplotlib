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
    os.system('cd build; rsync -avz html jdh2358@matplotlib.sf.net:/home/groups/m/ma/matplotlib/htdocs/doc/ -essh')

def sfpdf():
    'push a copy to the sf site'
    os.system('cd build/latex; scp Matplotlib.pdf jdh2358@matplotlib.sf.net:/home/groups/m/ma/matplotlib/htdocs/doc/')

def figs():
    os.system('cd users/figures/ && python make.py')

def html():
    check_build()
    #figs()
    if os.system('sphinx-build -b html -d build/doctrees . build/html'):
        raise SystemExit("Building HTML failed.")

    figures_dest_path = 'build/html/pyplots'
    if os.path.exists(figures_dest_path):
        shutil.rmtree(figures_dest_path)
    shutil.copytree('pyplots', figures_dest_path)

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
    if os.path.exists('build'):
        shutil.rmtree('build')
    for fname in glob.glob('pyplots/*.png') + glob.glob('pyplots/*.pdf'):
        os.remove(fname)

def all():
    #figs()
    html()
    latex()


funcd = {'figs':figs,
         'html':html,
         'latex':latex,
         'clean':clean,
         'sf':sf,
         'sfpdf':sfpdf,
         'all':all,
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
