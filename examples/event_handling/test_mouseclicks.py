#!/usr/bin/env python
import matplotlib
#matplotlib.use("WxAgg")
#matplotlib.use("TkAgg")
matplotlib.use("GTKAgg")
#matplotlib.use("QtAgg")
#matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import math
import numpy

print "***** TESTING WITH BACKEND: %s"%matplotlib.get_backend() + " *****"

def OnClick(event):
   if event.dblclick:
      print "DBLCLICK",event
   else:
      print "DOWN",event

def OnRelease(event):
    print "UP",event


fig = plt.gcf()
cid_up = fig.canvas.mpl_connect('button_press_event',OnClick)
cid_down = fig.canvas.mpl_connect('button_release_event',OnRelease)

t = numpy.arange(0.0,3.0,0.01)
s = numpy.sin(2*math.pi*t)
plt.plot(t,s)

plt.show()

