#!/usr/bin/env python

import gobject
import gtk

from pylab import *


fig = figure(1)
ind = arange(30)
X = rand(len(ind),10)
lines = plot(X[:,0], 'o')

manager = get_current_fig_manager()
def updatefig(*args):
    lines[0].set_data(ind, X[:,updatefig.count])
    manager.canvas.draw()
    updatefig.count += 1
    if updatefig.count<10:
        return True
    else:
        return False

updatefig.count = 0

gobject.timeout_add(300, updatefig)
show()
