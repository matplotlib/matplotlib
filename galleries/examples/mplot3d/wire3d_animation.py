"""
===========================
Animate a 3D wireframe plot
===========================

A very simple "animation" of a 3D plot.  See also :doc:`rotate_axes3d_sgskip`.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

FRAMES = 25
FPS = 25

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 50)
ys = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(xs, ys)

# Set the z axis limits, so they aren't recalculated each frame.
ax.set_zlim(-1, 1)

# Begin plotting.
wframe = None
tstart = time.time()


def animate(i):
    global wframe
    # If a line collection is already there, remove it before drawing.
    if wframe:
        wframe.remove()
    # Generate data.
    phi = i / FRAMES * 2 * np.pi
    Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))
    # Plot the new wireframe.
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    if i == FRAMES - 1:  # Print FPS at the end of the loop.
        global tstart
        fps = FRAMES / (time.time() - tstart)
        print(f'Expected FPS: {FPS}; Average FPS: {fps}')
        tstart = time.time()


ani = animation.FuncAnimation(fig, animate, interval=1000 / FPS, frames=FRAMES)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    component: animation,
#    level: beginner
