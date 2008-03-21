#!/usr/bin/env python
import sys, os, glob
import matplotlib
matplotlib.use('Agg')

def figs():
    # each one of these will make a figure when imported
    import dollar_ticks
    import fig_axes_customize_simple
    import fig_axes_labels_simple

    print 'all figures made'
    for fname in glob.glob('*.pyc'):
        os.remove(fname)

def clean():
    patterns = ['#*', '*~', '*.png']
    for pattern in patterns:
        for fname in glob.glob(pattern):
            os.remove(fname)
    print 'all clean'



def all():
    figs()

funcd = {'figs':figs,
         'clean':clean,
         'all':all,
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




