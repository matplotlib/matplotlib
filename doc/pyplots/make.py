#!/usr/bin/env python
import sys, os, glob
import matplotlib
import IPython.Shell
#matplotlib.rcdefaults()
matplotlib.use('Agg')

mplshell = IPython.Shell.MatplotlibShell('mpl')

formats = [('png', 100),
           ('hires.png', 200),
           ('pdf', 72)]

def figs():
    print 'making figs'
    import matplotlib.pyplot as plt
    for fname in glob.glob('*.py'):
        if fname==__file__: continue
        basename, ext = os.path.splitext(fname)
        outfiles = ['%s.%s' % (basename, format) for format, dpi in formats]
        all_exists = True
        for format, dpi in formats:
            if not os.path.exists('%s.%s' % (basename, format)):
                all_exists = False
                break

        if all_exists:
            print '    already have %s'%fname
        else:
            print '    building %s'%fname
            plt.close('all')    # we need to clear between runs
            mplshell.magic_run(basename)
            for format, dpi in formats:
                plt.savefig('%s.%s' % (basename, format), dpi=dpi)
    print 'all figures made'


def clean():
    patterns = (['#*', '*~', '*pyc'] +
                ['*.%s' % format for format, dpi in formats])
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




