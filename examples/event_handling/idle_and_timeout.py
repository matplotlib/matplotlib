from __future__ import print_function
"""
Demonstrate/test the idle and timeout API

This is only tested on gtk so far and is a prototype implementation
"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

t = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2*np.pi*t)
y2 = np.cos(2*np.pi*t)
line1, = ax.plot(y1)
line2, = ax.plot(y2)

N = 100
def on_idle(event):
    on_idle.count +=1
    print('idle', on_idle.count)
    line1.set_ydata(np.sin(2*np.pi*t*(N-on_idle.count)/float(N)))
    event.canvas.draw()
    # test boolean return removal
    if on_idle.count==N:
        return False
    return True
on_idle.cid = None
on_idle.count = 0

fig.canvas.mpl_connect('idle_event', on_idle)

plt.show()


