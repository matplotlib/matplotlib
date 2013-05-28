#!/usr/bin/env python

"""
A simple example of an animated plot in tkagg
"""
from __future__ import print_function
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

def animate():
    tstart = time.time()                   # for profiling
    x = np.arange(0, 2*np.pi, 0.01)        # x-array
    line, = ax.plot(x, np.sin(x))

    for i in np.arange(1,200):
        line.set_ydata(np.sin(x+i/10.0))          # update the data
        fig.canvas.draw()                         # redraw the canvas
    print('FPS:' , 200/(time.time()-tstart))

win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()
