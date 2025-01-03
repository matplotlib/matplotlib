"""
======================
Artists as annotations
======================

`.AnnotationBbox` creates an annotation using an `.OffsetBox` object, which is a class
of container artists for positioning an artist relative to a parent artist. This allows
for annotations that are texts, images, and arbitrary artists.  `.AnnotationBbox` also
provides more fine-grained control than `.Axes.annotate`.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from matplotlib.patches import Annulus, Circle

# %%%%
# Text
# ====
#
# `.AnnotationBbox` supports positioning annotations relative to data, Artists, and
# callables, as described in :ref:`annotations`. The  `.TextArea` is used to create a
# textbox that is not explicitly attached to an axes.
#
fig, ax = plt.subplots()

# Define a 1st position to annotate (display it with a marker)
xy = (0.5, 0.7)
ax.plot(xy[0], xy[1], ".r")

# Annotate the 1st position with a text box ('Test 1')
offsetbox = TextArea("Test 1")

ab1 = AnnotationBbox(offsetbox, xy,
                     xybox=(-20, 40),
                     xycoords='data',
                     boxcoords="offset points",
                     arrowprops=dict(arrowstyle="->"),
                     bboxprops=dict(boxstyle="sawtooth"))
ax.add_artist(ab1)

# Annotate the 1st position with another text box ('Test')
offsetbox = TextArea("Test 2")

ab2 = AnnotationBbox(offsetbox, (1, .85),
                     xybox=(.75, xy[1]),
                     xycoords=ab1,
                     boxcoords=("axes fraction", "data"),
                     box_alignment=(0., 0.5),
                     arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab2)

# %%%%
# Images
# ======
# The `.OffsetImage` container supports plotting images using `.BboxImage`

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
# ================
# `.DrawingArea` artists position arbitrary artists relative to their parent artists.

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
