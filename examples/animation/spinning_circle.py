"""
=========================
Spinning circle animation
=========================

This example shows how to create an animated spinning circle loading icon.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

ax.axis('off')

lines = ax.vlines(
    np.radians(np.arange(0, 360, 30)),
    0.5, 
    1, 
    lw=17,
    colors=np.linspace(0.15, 0.85, 12, dtype=str),
    capstyle='round',
)

ax.set_rmax(1.1)  # make round capstyle on outside visible

def update(*args):
    new_colors = np.roll(lines.get_colors(), shift=-1, axis=0)
    lines.set_color(new_colors)
    return lines,


ani = animation.FuncAnimation(
    fig, 
    update,
    interval=100)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.vlines` / `matplotlib.pyplot.vlines`
#    - `matplotlib.animation.FuncAnimation`
