# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations

from __future__ import print_function

import os
import sys

#import matplotlib
#matplotlib.use('Qt4Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4 import QtCore, QtGui

ITERS = 1000

import numpy as np
import time

class BlitQT(FigureCanvas):

    def __init__(self):
        FigureCanvas.__init__(self, Figure())

        self.ax = self.figure.add_subplot(111)
        self.ax.grid()
        self.draw()

        self.old_size = self.ax.bbox.width, self.ax.bbox.height
        self.ax_background = self.copy_from_bbox(self.ax.bbox)
        self.cnt = 0

        self.x = np.arange(0,2*np.pi,0.01)
        self.sin_line, = self.ax.plot(self.x, np.sin(self.x), animated=True)
        self.cos_line, = self.ax.plot(self.x, np.cos(self.x), animated=True)
        self.draw()

        self.tstart = time.time()
        self.startTimer(10)

    def timerEvent(self, evt):
        current_size = self.ax.bbox.width, self.ax.bbox.height
        if self.old_size != current_size:
            self.old_size = current_size
            self.ax.clear()
            self.ax.grid()
            self.draw()
            self.ax_background = self.copy_from_bbox(self.ax.bbox)

        self.restore_region(self.ax_background)

        # update the data
        self.sin_line.set_ydata(np.sin(self.x+self.cnt/10.0))
        self.cos_line.set_ydata(np.cos(self.x+self.cnt/10.0))
        # just draw the animated artist
        self.ax.draw_artist(self.sin_line)
        self.ax.draw_artist(self.cos_line)
        # just redraw the axes rectangle
        self.blit(self.ax.bbox)

        if self.cnt == 0:
            # TODO: this shouldn't be necessary, but if it is excluded the
            # canvas outside the axes is not initially painted.
            self.draw()
        if self.cnt==ITERS:
            # print the timing info and quit
            print('FPS:' , ITERS/(time.time()-self.tstart))
            sys.exit()
        else:
            self.cnt += 1

app = QtGui.QApplication(sys.argv)
widget = BlitQT()
widget.show()

sys.exit(app.exec_())
