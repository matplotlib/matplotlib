"""
===========================
Animate a 3D wireframe plot
===========================

A very simple "animation" of a 3D plot.  See also :doc:`rotate_axes3d_sgskip`.
"""

# sphinx_gallery_thumbnail_path = "_static/wire3d_animation.png"
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

# Plot the first frame. Remove the line so we can draw the next frame.
Z = np.sin(2 * np.pi * (X + Y) * 0.5)
wframe = ax.plot_wireframe(X, Y, Z, color="C0", linewidth=2)
wframe.remove()

t0 = time.time()
for i in range(FRAMES):
    # Compute and draw a new frame
    Z = np.sin(2 * np.pi * (X + Y) * (i + 1) / FRAMES)
    wframe = ax.plot_wireframe(X, Y, Z, color="C0", linewidth=2)
    # Make sure the background is drawn first (to remove the previous wireframe)
    if i > 1:
        ax.collections[-2].remove()
    # Need to draw and flush the events to show the frame
    fig.canvas.draw()
    # A short pause to show the frame
    time.sleep(1 / FPS)

t1 = time.time()
print(f"Frames per second: {FRAMES / (t1 - t0):0.1f}")
