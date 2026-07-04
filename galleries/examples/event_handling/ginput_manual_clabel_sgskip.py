"""
=====================
Interactive functions
=====================

This provides examples of uses of interactive functions, such as ginput,
waitforbuttonpress and manual clabel placement.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# sphinx_gallery_thumbnail_path = "_static/ginput_manual_clabel.png"
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
fig, ax = plt.subplots()
ax.plot(t, s)

np.random.seed(19680801)


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

# Set up a figure, wait for input, and highlight the selected point.

tellme('Click to begin')

plt.waitforbuttonpress()

while True:
    pts = []
    tellme('Select two corners of zoom, enter to start')
    pts = np.asarray(plt.ginput(2, timeout=-1))
    if len(pts) < 2:
        break
    (x0, y0), (x1, y1) = pts
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.draw()
    tellme('Click when ready')
    plt.waitforbuttonpress()

tellme('All done')
plt.show()
