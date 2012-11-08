#!/usr/bin/env python

from __future__ import print_function

# For detailed comments on animation and the techniques used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import pipong


fig, ax = plt.subplots()
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
print('FPS:' , animation.cnt/(time.time()-tstart))
