# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations

from __future__ import print_function

import sys
import matplotlib
matplotlib.use('QtAgg') # qt3 example

from qt import *
# Note: color-intensive applications may require a different color allocation
# strategy.
QApplication.setColorSpec(QApplication.NormalColor)

TRUE  = 1
FALSE = 0
ITERS = 1000

import matplotlib.pyplot as plt
import numpy as npy
import time

class BlitQT(QObject):
    def __init__(self):
        QObject.__init__(self, None, "app")

        self.ax = plt.subplot()
        self.canvas = self.ax.figure.canvas
        self.cnt = 0

        # create the initial line
        self.x = npy.arange(0,2*npy.pi,0.01)
        self.line, = plt.plot(self.x, npy.sin(self.x), animated=True, lw=2)

        self.background = None

    def timerEvent(self, evt):
        if self.background is None:
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
            print('FPS:', ITERS/(time.time()-self.tstart))
            sys.exit()

        else:
            self.cnt += 1

plt.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
plt.grid() # to ensure proper background restore

app = BlitQT()
# for profiling
app.tstart = time.time()
app.startTimer(0)

plt.show()
