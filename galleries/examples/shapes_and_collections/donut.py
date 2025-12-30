r"""
===================================
Draw Donuts with Path and PathPatch
===================================

Mmh... draw donuts (miam!) using `~.path.Path`\s and `~.patches.PathPatch`\es.
This example shows the effect of the path's orientations in a compound path.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.path as mpath


def wise(v):
    return {+1: "CCW", -1: "CW"}[v]


def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack((x, y))


fig, ax = plt.subplots()

inside_vertices = make_circle(0.5)
outside_vertices = make_circle(1.0)
codes = np.full(len(inside_vertices), mpath.Path.LINETO)
codes[0] = mpath.Path.MOVETO

for i, (inside, outside) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    # Concatenate the inside and outside subpaths together, changing their
    # order as needed
    vertices = np.concatenate((outside_vertices[::outside],
                               inside_vertices[::inside]))
    # Shift the path
    vertices += (i * 2.5, 0)
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    all_codes = np.concatenate((codes, codes))
    # Create the Path object
    path = mpath.Path(vertices, all_codes)
    # And plot it
    patch = mpatches.PathPatch(path, facecolor='#885500', edgecolor='black')
    ax.add_patch(patch)

    ax.annotate(f"Outside {wise(outside)},\nInside {wise(inside)}",
                (i * 2.5, -1.5), va="top", ha="center")

ax.set(xlim=(-2, 10), ylim=(-3, 2), aspect=1, title="Mmm, donuts!")
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.path`
#    - `matplotlib.path.Path`
#    - `matplotlib.patches`
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.patches.Circle`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.axes.Axes.annotate`
#    - `matplotlib.axes.Axes.set_aspect`
#    - `matplotlib.axes.Axes.set_xlim`
#    - `matplotlib.axes.Axes.set_ylim`
#    - `matplotlib.axes.Axes.set_title`
#
# .. tags::
#
#    component: patch
#    styling: shape
#    purpose: fun
