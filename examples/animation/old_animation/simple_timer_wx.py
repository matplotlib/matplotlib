from __future__ import print_function
"""
A simple example of an animated plot using a wx backend
"""
import numpy as np
import matplotlib
matplotlib.use('WXAgg') # do this before importing pylab

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.arange(0, 2*np.pi, 0.1)
line, = ax.plot(t, np.sin(t))
dt = 0.05


def update_line(event):
    if update_line.i==200:
        return False
    print('update', update_line.i)
    line.set_ydata(np.sin(t+update_line.i/10.))
    fig.canvas.draw()                 # redraw the canvas
    update_line.i += 1
update_line.i = 0

import wx
id = wx.NewId()
actor = fig.canvas.manager.frame
timer = wx.Timer(actor, id=id)
timer.Start(100)
wx.EVT_TIMER(actor, id, update_line)
#actor.Bind(wx.EVT_TIMER, update_line, id=id)

plt.show()
