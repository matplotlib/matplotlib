"""
===================================================
Select open polygonal chains using polygon selector
===================================================

`.widgets.PolygonSelector` is used to select closed polygonal regions.
It can also be used to select open polygonal chains by disabling closure
with ``closed=False``.

This example demonstrates how to create open polygonal chains using
`~.widgets.PolygonSelector` both programmatically and interactively,
and how repeated selections can be used to draw multiple lines.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import PolygonSelector

# %%
#
# To create the open polygonal chain programmatically

fig1, ax1 = plt.subplots()
fig1.show()

selector1 = PolygonSelector(ax1, closed=False)

selector1.verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]

# %%
#
# To create the open polygonal chain interactively


def print_interactive_usage():
    print("Click to add vertices sequentially.")
    print("Press the 'enter' key to complete the open polygonal chain.")
    print("Hold 'ctrl' to reposition a single vertex while chain is incomplete.")
    print("Hold 'shift' to move all vertices.")
    print("Left click and drag a vertex to reposition it.")
    print("Right click to remove a vertex.")
    print("Press the 'esc' key to start a new open polygonal chain.")

fig2, ax2 = plt.subplots()
fig2.show()

selector2 = PolygonSelector(ax2, closed=False)

print_interactive_usage()

# %%
#
# Repeatedly create open polygonal chains interactively
# to draw multiple lines within the same axes

fig3, ax3 = plt.subplots()
fig3.show()

ax3.set(xlim=(0, 10), ylim=(0, 10))
ax3.grid(alpha=0.5)


def onselect(verts):
    x, y = zip(*verts)
    ax3.plot(x, y)
    print("Chain completed. Press 'esc' to start a new one.")

selector3 = PolygonSelector(ax3, onselect, closed=False)

print_interactive_usage()

# %%
#
# Extending the previous example, it can be used in simple applications
# such as constructing a graph structure, where vertices and edges are defined
# interactively by repeatedly selecting open polygonal chains.

graph_verts = []
snap_range = 0.5


def snap_vertex(vertex):
    """Snap chain vertex to an existing graph vertex if within range."""
    vertex = np.asarray(vertex)

    for v in graph_verts:
        dist = np.linalg.norm(vertex - v)
        if dist < snap_range:
            return v

    graph_verts.append(vertex)
    return vertex

fig4, ax4 = plt.subplots()
fig4.show()

ax4.set(xlim=(0, 10), ylim=(0, 10))
ax4.set_xticks([])
ax4.set_yticks([])


def onselect(verts):
    snapped = [snap_vertex(v) for v in verts]
    x, y = zip(*snapped)

    ax4.plot(x, y, color='black', linewidth=3, marker='o', markersize=12,
             markerfacecolor="#8888FF", markeredgecolor='black')

    print("Chain completed. Press 'esc' to start a new one.")

selector4 = PolygonSelector(ax4, onselect, closed=False)

print_interactive_usage()


# %%
# .. tags::
#
#    component: axes,
#    styling: position,
#    plot-type: line,
#    level: intermediate,
#    domain: cartography,
#    domain: geometry,
#    domain: statistics,
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.PolygonSelector`
