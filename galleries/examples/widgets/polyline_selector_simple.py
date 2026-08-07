"""
=================
Polyline Selector
=================

Shows how to create a polyline programmatically or interactively
"""

import matplotlib.pyplot as plt

from matplotlib.widgets import PolylineSelector

# %%
#
# To create the polyline programmatically
fig, ax = plt.subplots()
fig.show()

selector = PolylineSelector(ax, lambda *args: None)

# Add three vertices
selector.verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]


# %%
#
# To create the polyline interactively

fig2, ax2 = plt.subplots()
fig2.show()

selector2 = PolylineSelector(ax2, lambda *args: None)

print("Click to add vertices sequentially.")
print("Press the 'enter' key to complete the polyline.")
print("Hold 'ctrl' to reposition a single vertex while polyline is incomplete.")
print("Hold 'shift' to move all vertices.")
print("Left click and drag a vertex to reposition it.")
print("Right click to remove a vertex.")
print("Press the 'esc' key to start a new polyline.")


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
#    - `matplotlib.widgets.PolylineSelector`
