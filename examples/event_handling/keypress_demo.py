#!/usr/bin/env python

"""
Show how to connect to keypress events
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def press(event):
    print('press', event.key)
    if event.key=='x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()

fig = plt.figure()
ax = fig.add_subplot(111)

fig.canvas.mpl_connect('key_press_event', press)

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')

plt.show()
