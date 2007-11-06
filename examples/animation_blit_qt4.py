# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations

import os, sys
import matplotlib
matplotlib.use('Qt4Agg') # qt4 example

from PyQt4 import QtCore, QtGui

ITERS = 1000

import pylab as p
import numpy as npy
import time

class BlitQT(QtCore.QObject):
    def __init__(self):
        self.ax = p.subplot(111)
        self.canvas = self.ax.figure.canvas

        # By making this a child of the canvas we make sure that it is
        # destroyed first and avoids a possible exception when the user clicks
        # on the window's close box.
        QtCore.QObject.__init__(self, self.canvas)
        
        self.cnt = 0

        # create the initial line
        self.x = npy.arange(0,2*npy.pi,0.01)
        self.line, = p.plot(self.x, npy.sin(self.x), animated=True, lw=2)

        self.background = None
        self.old_size = 0, 0

    def timerEvent(self, evt):
        # See if the size has changed since last time round.
        current_size = self.ax.bbox.width(), self.ax.bbox.height()

        if self.old_size != current_size:
            self.old_size = current_size
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # restore the clean slate background
        self.canvas.restore_region(self.background)
        # update the data
        self.line.set_ydata(npy.sin(self.x+self.cnt/10.0))
        # just draw the animated artist
        self.ax.draw_artist(self.line)
        # just redraw the axes rectangle
        self.canvas.blit(self.ax.bbox)

        if self.cnt==ITERS:
            # print the timing info and quit
            print 'FPS:' , ITERS/(time.time()-self.tstart)
            sys.exit()

        else:
            self.cnt += 1

p.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
p.grid() # to ensure proper background restore

app = BlitQT()
# for profiling
app.tstart = time.time()
app.startTimer(0)

p.show()
