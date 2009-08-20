#!/usr/bin/env python

# For detailed comments on animation and the techniques used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import numpy as np
import matplotlib.pyplot as plt
import pipong
from numpy.random import randn, randint


fig = plt.figure()
ax = fig.add_subplot(111)
canvas = ax.figure.canvas


def start_anim(event):
#    gobject.idle_add(animation.draw,animation)
    gobject.timeout_add(10,animation.draw,animation)
    canvas.mpl_disconnect(start_anim.cid)

animation = pipong.Game(ax)
start_anim.cid = canvas.mpl_connect('draw_event', start_anim)


tstart = time.time()
plt.grid() # to ensure proper background restore
plt.show()
print 'FPS:' , animation.cnt/(time.time()-tstart)
