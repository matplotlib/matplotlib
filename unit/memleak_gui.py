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
parser.add_option("-c", "--cycles", dest="cycles",
                  default=False, action="store_true")

options, args = parser.parse_args()

indStart = int(options.start)
indEnd = int(options.end)
import matplotlib
matplotlib.rcParams['toolbar'] = matplotlib.validate_toolbar(options.toolbar)
if options.backend:
    matplotlib.use(options.backend)
import pylab
import matplotlib.cbook as cbook

print '# columns are: iteration, OS memory (k), number of python objects'
print '#'
for i in range(indEnd+1):

    fig = pylab.figure()
    fig.savefig('test')  # This seems to just slow down the testing.
    fig.clf()
    pylab.close(fig)
    gc.collect()
    val = cbook.report_memory(i)
    if options.verbose:
        if i % 10 == 0:
            #print ("iter: %4d OS memory: %8d Python objects: %8d" %
            print ("%4d %8d %8d" %
                   (i, val, len(gc.get_objects())))
    if i==indStart: start = val # wait a few cycles for memory usage to stabilize

gc.collect()
end = val

print '# columns above are: iteration, OS memory (k), number of python objects'
print '#'
print '# uncollectable list:', gc.garbage
print '#'

if i > indStart:
    print '# Backend %(backend)s, toolbar %(toolbar)s' % matplotlib.rcParams
    backend = options.backend.lower()
    if backend.startswith("gtk"):
        import gtk
        import gobject
        print "# pygtk version: %s, gtk version: %s, pygobject version: %s, glib version: %s" % \
            (gtk.pygtk_version, gtk.gtk_version, 
             gobject.pygobject_version, gobject.glib_version)
    elif backend.startswith("qt4"):
        import PyQt4.pyqtconfig
        print "# PyQt4 version: %s, Qt version %x" % \
            (PyQt4.pyqtconfig.Configuration().pyqt_version_str,
             PyQt4.pyqtconfig.Configuration().qt_version)
    elif backend.startswith("qt"):
        import pyqtconfig
        print "# pyqt version: %s, qt version: %x" % \
            (pyqtconfig.Configuration().pyqt_version_str,
             pyqtconfig.Configuration().qt_version)
    elif backend.startswith("wx"):
        import wx
        print "# wxPython version: %s" % wx.__version__
    elif backend.startswith("tk"):
        import Tkinter
        print "# Tkinter version: %s, Tk version: %s, Tcl version: %s" % (Tkinter.__version__, Tkinter.TkVersion, Tkinter.TclVersion)

    print '# Averaging over loops %d to %d' % (indStart, indEnd)
    print '# Memory went from %dk to %dk' % (start, end)
    print '# Average memory consumed per loop: %1.4fk bytes\n' % ((end-start)/float(indEnd-indStart))

if options.cycles:
    cbook.print_cycles(gc.garbage)
