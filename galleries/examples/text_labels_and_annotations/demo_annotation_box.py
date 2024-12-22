"""
======================
Artists as annotations
======================

`.AnnotationBbox` facilitates annotating parts of the figure or axes using arbitrary
artists, such as texts, images, and `matplotlib.patches`. `.AnnotationBbox` supports
these artists via inputs that are subclasses of `.OffsetBox`, which is a class of
container artists for positioning an artist relative to a parent artist.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage, TextArea
from matplotlib.patches import Annulus, Circle, ConnectionPatch

# %%%%
# Text
# ====
#
# `.AnnotationBbox` supports positioning annotations relative to data, Artists, and
# callables, as described in :ref:`annotations`. The  `.TextArea` is used to create a
# textbox that is not explicitly attached to an axes, which allows it to be used for
# annotating figure objects. The `Axes.annotate` method should be used when annotating
# an axes element (such as a plot) with text.
#
fig, axd = plt.subplot_mosaic([['t1', '.', 't2']], layout='compressed')

# Define a 1st position to annotate (display it with a marker)
xy1 = (.25, .75)
xy2 = (.75, .25)
axd['t1'].plot(*xy1, ".r")
axd['t2'].plot(*xy2, ".r")
axd['t1'].set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
axd['t2'].set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
# Draw an arrow between the points

c = ConnectionPatch(xyA=xy1, xyB=xy2,
                    coordsA=axd['t1'].transData, coordsB=axd['t2'].transData,
                    arrowstyle='->')
fig.add_artist(c)

# Annotate the ConnectionPatch position ('Test 1')
offsetbox = TextArea("Test 1")

ab1 = AnnotationBbox(offsetbox, (.5, .5),
                     xybox=(0, 0),
                     xycoords=c,
                     boxcoords="offset points",
                     arrowprops=dict(arrowstyle="->"),
                     bboxprops=dict(boxstyle="sawtooth"))
fig.add_artist(ab1)

# %%%%
# Images
# ======
# The `.OffsetImage` container facilitates using images as annotations


fig, ax = plt.subplots()
# Define a position to annotate (don't display with a marker)
xy = [0.3, 0.55]

# Annotate a position with an image generated from an array of pixels
arr = np.arange(100).reshape((10, 10))
im = OffsetImage(arr, zoom=2, cmap='viridis')
im.image.axes = ax

ab = AnnotationBbox(im, xy,
                    xybox=(-50., 50.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.3,
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)

# Annotate the position with another image (a Grace Hopper portrait)
with get_sample_data("grace_hopper.jpg") as file:
    arr_img = plt.imread(file)

imagebox = OffsetImage(arr_img, zoom=0.2)
imagebox.image.axes = ax

ab = AnnotationBbox(imagebox, xy,
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )

ax.add_artist(ab)

# Fix the display limits to see everything
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.show()

# %%%%
# Arbitrary Artists
# =================
#
# Multiple and arbitrary artists can be placed inside a `.DrawingArea`.


fig, ax = plt.subplots()
# Define a position to annotate (don't display with a marker)
xy = [0.3, 0.55]

# Annotate the position with a circle and annulus
da = DrawingArea(30, 30, 0, 0)
p = Circle((10, 10), 10, color='C0')
da.add_artist(p)
q = Annulus((20, 20), 10, 5, color='C1')
da.add_artist(q)


# Use the drawing area as an annotation
ab = AnnotationBbox(da, xy,
                    xybox=(.75, xy[1]),
                    xycoords='data',
                    boxcoords=("axes fraction", "data"),
                    box_alignment=(0.2, 0.5),
                    arrowprops=dict(arrowstyle="->"),
                    bboxprops=dict(alpha=0.5))

ax.add_artist(ab)
plt.show()
#

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.Circle`
#    - `matplotlib.offsetbox.TextArea`
#    - `matplotlib.offsetbox.DrawingArea`
#    - `matplotlib.offsetbox.OffsetImage`
#    - `matplotlib.offsetbox.AnnotationBbox`
#    - `matplotlib.cbook.get_sample_data`
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.pyplot.imread`
#
# .. tags::
#    component: annotation, styling: position
