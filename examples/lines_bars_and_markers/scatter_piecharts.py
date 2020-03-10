"""
===================================
Scatter plot with pie chart markers
===================================

This example shows two methods to make custom 'pie charts' as the markers
for a scatter plot.
"""

##########################################################################
# Manually creating marker vertices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

import numpy as np
import matplotlib.pyplot as plt

# first define the cumulative ratios
r1 = 0.2       # 20%
r2 = r1 + 0.4  # 40%

# define some sizes of the scatter marker
sizes = np.array([60, 80, 120])

# calculate the points of the first pie marker
# these are just the origin (0, 0) + some (cos, sin) points on a circle
x1 = np.cos(2 * np.pi * np.linspace(0, r1))
y1 = np.sin(2 * np.pi * np.linspace(0, r1))
xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
s1 = np.abs(xy1).max()

x2 = np.cos(2 * np.pi * np.linspace(r1, r2))
y2 = np.sin(2 * np.pi * np.linspace(r1, r2))
xy2 = np.row_stack([[0, 0], np.column_stack([x2, y2])])
s2 = np.abs(xy2).max()

x3 = np.cos(2 * np.pi * np.linspace(r2, 1))
y3 = np.sin(2 * np.pi * np.linspace(r2, 1))
xy3 = np.row_stack([[0, 0], np.column_stack([x3, y3])])
s3 = np.abs(xy3).max()

fig, ax = plt.subplots()
ax.scatter(range(3), range(3), marker=xy1, s=s1**2 * sizes, facecolor='C0')
ax.scatter(range(3), range(3), marker=xy2, s=s2**2 * sizes, facecolor='C1')
ax.scatter(range(3), range(3), marker=xy3, s=s3**2 * sizes, facecolor='C2')

plt.show()


##########################################################################
# Using wedges as markers
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# An alternative is to create custom markers from the `~.path.Path` of a
# `~.patches.Wedge`, which might be more versatile.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.markers import MarkerStyle

# first define the ratios
r1 = 0.2          # 20%
r2 = r1 + 0.3     # 50%
r3 = 1 - r1 - r2  # 30%


def markers_from_ratios(ratios, width=1):
    markers = []
    angles = 360*np.concatenate(([0], np.cumsum(ratios)))
    for i in range(len(angles)-1):
        # create a Wedge within the unit square in between the given angles...
        w = Wedge((0, 0), 0.5, angles[i], angles[i+1], width=width/2)
        # ... and create a custom Marker from its path.
        markers.append(MarkerStyle(w.get_path(), normalization="none"))
    return markers

# define some sizes of the scatter marker
sizes = np.array([100, 200, 400, 800])
# collect the markers and some colors
markers = markers_from_ratios([r1, r2, r3], width=0.6)
colors = plt.cm.tab10.colors[:len(markers)]

fig, ax = plt.subplots()

for marker, color in zip(markers, colors):
    ax.scatter(range(len(sizes)), range(len(sizes)), marker=marker, s=sizes,
               edgecolor="none", facecolor=color)

ax.margins(0.1)
plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.axes.Axes.scatter
matplotlib.pyplot.scatter
matplotlib.patches.Wedge
matplotlib.markers.MarkerStyle
