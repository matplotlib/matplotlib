"""
A simple example of an animated plot using a wx backend
"""
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('WXAgg') # do this before importing pylab

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.arange(0, 2*np.pi, 0.1)
line, = ax.plot(t, np.sin(t))
dt = 0.05

def update_line(idleevent):
    if update_line.i==200:
        return False
    print('animate', update_line.i)
    line.set_ydata(np.sin(t+update_line.i/10.))
    fig.canvas.draw_idle()                 # redraw the canvas
    update_line.i += 1
update_line.i = 0

import wx
wx.EVT_IDLE(wx.GetApp(), update_line)
plt.show()
