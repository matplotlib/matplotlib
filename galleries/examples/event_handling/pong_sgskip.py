"""
====
Pong
====

A Matplotlib based game of Pong illustrating one way to write interactive
animations that are easily ported to multiple backends.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# sphinx_gallery_thumbnail_path = "_static/pong.png"
import time

import matplotlib.pyplot as plt
import numpy as np


class Player:
    ...


class Ball:
    ...


def init():
    line.set_data([], [])
    return line,


def animate(timer):
    ...


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

plt.show()
