"""
Emulate an oscilloscope.  Requires the animation API introduced in
matplotlib 0.84.  See
http://www.scipy.org/wikis/topical_software/Animations for an
explanation.

This example uses gtk but does not depend on it intimately.  It just
uses the idle handler to trigger events.  You can plug this into a
different GUI that supports animation (GTKAgg, TkAgg, WXAgg) and use
your toolkits idle/timer functions.
"""
import gobject
import matplotlib
matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Scope:
    def __init__(self, ax, maxt=10, dt=0.01):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata, animated=True)
        self.ax.add_line(self.line)
        self.background = None
        self.canvas.mpl_connect('draw_event', self.update_background)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

    def update_background(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def emitter(self, p=0.01):
        'return a random value with probability p, else 0'
        v = np.random.rand(1)
        if v>p: return 0.
        else: return np.random.rand(1)

    def update(self, *args):
        if self.background is None: return True
        y = self.emitter()
        lastt = self.tdata[-1]
        if lastt>self.tdata[0]+self.maxt: # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0]+self.maxt)
            self.ax.figure.canvas.draw()

        self.canvas.restore_region(self.background)

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        self.ax.draw_artist(self.line)

        self.canvas.blit(self.ax.bbox)
        return True


fig, ax = plt.subplots()
scope = Scope(ax)
gobject.idle_add(scope.update)

plt.show()
