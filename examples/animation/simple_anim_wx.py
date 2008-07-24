"""
A simple example of an animated plot using a gtk backends
"""
import time
import numpy as np
import matplotlib
matplotlib.use('WXAgg') # do this before importing pylab

import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)

def animate(idleevent):
    tstart = time.time()                   # for profiling
    x = np.arange(0, 2*np.pi, 0.01)        # x-array
    line, = ax.plot(x, np.sin(x))

    for i in np.arange(1,200):
        line.set_ydata(np.sin(x+i/10.0))  # update the data
        fig.canvas.draw()                 # redraw the canvas
    print 'FPS:' , 200/(time.time()-tstart)
    raise SystemExit

# call the animation loop on idle
import wx
wx.EVT_IDLE(wx.GetApp(), animate)
plt.show()
