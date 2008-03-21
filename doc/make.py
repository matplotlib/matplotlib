#!/usr/bin/env python
import os, sys, glob

def check_png():
    if not len(glob.glob('figures/*.png')):
        raise SystemExit('No PNG files in figures dir; please run make.py in the figures directory first')

def check_rst2latex():
    sin, sout = os.popen2('which rst2latex')
    if not sout.read():
        raise SystemExit('Build requires rst2latex')

def check_pdflatex():
    sin, sout = os.popen2('which pdflatex')
    if not sout.read():
        raise SystemExit('Build requires pdflatex')




def artist_tut():
    check_png()
    check_rst2latex()
    check_pdflatex()
    os.system('rst2latex artist_api_tut.txt > artist_api_tut.tex')
    os.system('pdflatex  artist_api_tut.tex')


def event_tut():
    check_png()
    check_rst2latex()
    check_pdflatex()
    os.system('rst2latex event_handling_tut.txt > event_handling_tut.tex')
    os.system('pdflatex event_handling_tut.tex')

def clean():
    patterns = ['#*', '*~', '*.tex', '*.log', '*.out', '*.aux', '*.pdf']
    for pattern in patterns:
        for fname in glob.glob(pattern):
            os.remove(fname)
    print 'all clean'

def all():
    artist_tut()
    event_tut()

funcd = {'artist_tut': artist_tut,
         'event_tut': event_tut,
         'clean': clean,
         'all': all,
         }

if len(sys.argv)>1:
    for arg in sys.argv[1:]:
        func = funcd.get(arg)
        if func is None:
            raise SystemExit('Do not know how to handle %s; valid args are'%(
                    arg, funcd.keys()))
        func()
else:
    all()










