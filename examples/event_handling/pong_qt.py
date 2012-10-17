# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations

from __future__ import print_function

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
import time
import pipong

class BlitQT(QObject):
    def __init__(self):
        QObject.__init__(self, None, "app")

        self.ax = plt.subplot(111)
        self.animation = pipong.Game(self.ax)

    def timerEvent(self, evt):
       self.animation.draw(evt)

plt.grid() # to ensure proper background restore

app = BlitQT()
# for profiling
app.tstart = time.time()
app.startTimer(10)

plt.show()
print('FPS:' , app.animation.cnt/(time.time()-app.tstart))
