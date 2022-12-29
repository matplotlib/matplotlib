"""
=====
Decay
=====

This example showcases:
- using PlayerAnimation
- using blitting
- changing axes limits during an animation.
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import PlayerAnimation, FuncAnimation
import math

def data_gen():
    for cnt in itertools.count():
        yield cnt

def init(val):
    global line
    print(f"init with val: {val}")
    if not "line" in globals():
        line, = ax.plot([], [], lw=2)
    ax.grid()
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1 + math.floor(val / 10))
    line.set_data([], [])
    return [line]

fig, ax = plt.subplots()

def update_plot(i):
    # update the data
    xdata = np.linspace(-10, 10, 1000)
    ydata = np.sin(xdata + i*0.1)
    _, xmax = ax.get_xlim()
    new_xmax = 1 + math.floor(i / 10)
    if xmax != new_xmax:
        ax.set_xlim(0, new_xmax)
        ax.figure.canvas.draw_idle()
    line.set_data(xdata, ydata)

    return [line]

animation = PlayerAnimation(fig=fig, func=update_plot, init_func=init, interval=100, blit=True, valstep=0.5)
plt.show()
