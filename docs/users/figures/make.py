#!/usr/bin/env python
import sys, os, glob
import matplotlib
import IPython.Shell
matplotlib.rcdefaults()
matplotlib.use('Agg')

mplshell = IPython.Shell.MatplotlibShell('mpl')

def figs():
    print 'making figs'
    import matplotlib.pyplot as plt
    for fname in glob.glob('*.py'):
        if fname==__file__: continue
        basename, ext = os.path.splitext(fname)
        outfile = '%s.png'%basename

        if os.path.exists(outfile):
            print '    already have %s'%outfile
            continue
        else:
            print '    building %s'%fname
        plt.close('all')    # we need to clear between runs
        mplshell.magic_run(basename)
        plt.savefig('%s.png'%basename)
    print 'all figures made'


def clean():
    patterns = ['#*', '*~', '*.png', '*pyc']
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




