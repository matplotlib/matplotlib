#!/usr/bin/env python
"""
A simple example of an animated plot in matplotlib.  You can test the
speed of animation of various backends by running the script with the
'-dSomeBackend' flag

Here are some numbers from my system, where FPS is the frames rendered
per second

  TkAgg     20 FPS
  GTK       50 FPS
  GTKAgg    36 FPS
  GTKCairo  15 FPS
  WX        11 FPS
  WXAgg     27 FPS

"""
from pylab import *
import time

# turn interactive mode on for dynamic updates.  If you aren't in
# interactive mode, you'll need to use a GUI event handler/timer.
ion()  

tstart = time.time()              # for profiling
x = arange(0,2*pi,0.01)           # x-array
line, = plot(x,sin(x))
for i in arange(1,200):
    line.set_ydata(sin(x+i/10.0))  # update the data
    draw()                         # redraw the canvas

print 'FPS:' , 200/(time.time()-tstart)
