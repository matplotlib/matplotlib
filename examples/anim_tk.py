# deprecated - this example is no longer needed.  Follow the model of
# anim.py to use interaction = True to avoid all the cruft of timers,
# callbacks and the likes used here

#!/usr/bin/env python2.3

import matplotlib
matplotlib.use('TkAgg')
import pylab

#import Tkinter as Tk
import matplotlib.numerix as numerix
fig = pylab.figure(1)
ind = numerix.arange(60)



x_tmp=[]
for i in range(100):
    x_tmp.append(numerix.sin((ind+i)*numerix.pi/15.0))

X=numerix.array(x_tmp)


lines = pylab.plot(X[:,0],'o')

manager = pylab.get_current_fig_manager()

def updatefig(*args):
    updatefig.count += 1
    lines[0].set_ydata(X[:,updatefig.count%60])
    manager.canvas.draw()
    return updatefig.count
updatefig.count=-1

def run(*args):
    print 'called run'

    import time
    tstart = time.time()
    while 1:
        cnt = updatefig()
        if cnt==100: break
    print 'elapsed', 100.0/(time.time() - tstart)
    
import Tkinter as Tk
manager.window.after(10, run)
manager.show()
Tk.mainloop()
