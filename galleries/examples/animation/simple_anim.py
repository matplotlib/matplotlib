"""
========================
Basic animated line plot
========================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()

# %%
# Contours are updated the same way, with `.ContourSet.set_data`. Recontouring
# the existing artist is faster than removing the contour set and making a new
# one, and it keeps the contours in the same place in the draw order, which
# matters when blitting. The levels are not recomputed, so the colors mean
# the same thing in every frame.

fig, ax = plt.subplots()

X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))


def f(t):
    return np.sin(X + t) * np.cos(Y - t)


cs = ax.contour(X, Y, f(0), levels=np.linspace(-0.9, 0.9, 7))


def animate_contour(i):
    cs.set_data(X, Y, f(i / 25))  # update the data.
    return cs,


ani_contour = animation.FuncAnimation(
    fig, animate_contour, interval=20, blit=True, save_count=50)

plt.show()

# %%
#
# .. tags::
#    component: animation,
#    level: beginner
