"""
Demonstrate/test the idle and timeout API
"""
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

t = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2*np.pi*t)
y2 = np.cos(2*np.pi*t)
line1, = ax.plot(y1)
line2, = ax.plot(y2)

N = 100
def on_idle(event):
    on_idle.count +=1
    print 'idle', on_idle.count
    line1.set_ydata(np.sin(2*np.pi*t*(N-on_idle.count)/float(N)))
    event.canvas.draw()
    # test boolean return removal
    if on_idle.count==N:
        fig.canvas.mpl_disconnect(on_idle.cid)
on_idle.cid = None
on_idle.count = 0

def on_timeout(canvas):
    on_timeout.count +=1
    line2.set_ydata(np.cos(2*np.pi*t*(N-on_idle.count)/float(N)))
    line2.figure.canvas.draw()
    print 'timeout', on_timeout.count
    # test explicit removal
    if on_timeout.count==N:
        return False
    return True
on_timeout.count = 0

on_idle.cid = fig.canvas.mpl_connect('idle_event', on_idle)
#fig.canvas.mpl_timeout_add(100, on_timeout)

plt.show()


