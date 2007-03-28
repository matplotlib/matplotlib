#!/usr/bin/env python
'''
This illustrates a leak that occurs with any interactive backend.

Run with : python memleak_gui.py -dGTKAgg   # or TkAgg, etc..
'''
import os, sys, time
import gc
import matplotlib

#matplotlib.use('TkAgg') # or TkAgg or WxAgg or QtAgg or Gtk
matplotlib.rcParams['toolbar'] = 'toolbar2'   # None, classic, toolbar2
#matplotlib.rcParams['toolbar'] = None   # None, classic, toolbar2

import pylab
from matplotlib import _pylab_helpers as ph
import matplotlib.cbook as cbook

indStart, indEnd = 30, 50
for i in range(indEnd):

    fig = pylab.figure()
    fig.savefig('test')
    fig.clf()
    pylab.close(fig)
    val = cbook.report_memory(i)
    print i, val 
    gc.collect()
    if i==indStart: start = val # wait a few cycles for memory usage to stabilize

gc.collect()
print
print 'uncollectable list:', gc.garbage
print
end = val
if i > indStart:
    print 'Average memory consumed per loop: %1.4fk bytes\n' % ((end-start)/float(indEnd-indStart))

