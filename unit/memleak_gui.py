#!/usr/bin/env python
'''
This illustrates a leak that occurs with any interactive backend.
Run with :

  > python memleak_gui.py -dGTKAgg   # or TkAgg, etc..

use --help option to see all options

The default number of loops typically will not yield a stable
estimate--for that you may need many hundreds of loops and some patience.

You may need to edit cbook.report_memory to support your platform

'''
import os, sys, time
import gc
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-q", "--quiet", default=True,
                  action="store_false", dest="verbose")
parser.add_option("-s", "--start", dest="start",
                  default="30",
                  help="first index of averaging interval")
parser.add_option("-e", "--end", dest="end",
                  default="100",
                  help="last index of averaging interval")
parser.add_option("-t", "--toolbar", dest="toolbar",
                  default="toolbar2",
                  help="toolbar: None, classic, toolbar2")
# The following overrides matplotlib's version of the -d option
# uses it if found
parser.add_option("-d", "--backend", dest="backend",
                  default='',
                  help="backend")


options, args = parser.parse_args()

indStart = int(options.start)
indEnd = int(options.end)
import matplotlib
matplotlib.rcParams['toolbar'] = matplotlib.validate_toolbar(options.toolbar)
if options.backend:
    matplotlib.use(options.backend)
import pylab
import matplotlib.cbook as cbook

for i in range(indEnd+1):

    fig = pylab.figure()
    #fig.savefig('test')  # This seems to just slow down the testing.
    fig.clf()
    pylab.close(fig)
    gc.collect()
    val = cbook.report_memory(i)
    if options.verbose:
        if i % 10 == 0:
            print i, val
    if i==indStart: start = val # wait a few cycles for memory usage to stabilize

gc.collect()
end = val

print '#'
print '# uncollectable list:', gc.garbage
print '#'

if i > indStart:
    print '# Backend %(backend)s, toolbar %(toolbar)s' % matplotlib.rcParams
    print '# Averaging over loops %d to %d' % (indStart, indEnd)
    print '# Memory went from %dk to %dk' % (start, end)
    print '# Average memory consumed per loop: %1.4fk bytes\n' % ((end-start)/float(indEnd-indStart))

