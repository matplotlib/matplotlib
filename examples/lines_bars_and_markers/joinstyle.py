"""
==========================
Join styles and cap styles
==========================

This example demonstrates the available join styles and cap styles.

Both are used in `.Line2D` and various ``Collections`` from
`matplotlib.collections` as well as some functions that create these, e.g.
`~matplotlib.pyplot.plot`.


Join styles
===========

Join styles define how the connection between two line segments is drawn.

See the respective ``solid_joinstyle``, ``dash_joinstyle`` or ``joinstyle``
parameters.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_angle(ax, x, y, angle, style):
    phi = np.radians(angle)
    xx = [x + .5, x, x + .5*np.cos(phi)]
    yy = [y, y, y + .5*np.sin(phi)]
    ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
    ax.plot(xx, yy, lw=1, color='black')
    ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Join style')

for x, style in enumerate(['miter', 'round', 'bevel']):
    ax.text(x, 5, style)
    for y, angle in enumerate([20, 45, 60, 90, 120]):
        plot_angle(ax, x, y, angle, style)
        if x == 0:
            ax.text(-1.3, y, f'{angle} degrees')
ax.text(1, 4.7, '(default)')

ax.set_xlim(-1.5, 2.75)
ax.set_ylim(-.5, 5.5)
ax.set_axis_off()
plt.show()


#############################################################################
#
# Cap styles
# ==========
#
# Cap styles define how the the end of a line is drawn.
#
# See the respective ``solid_capstyle``, ``dash_capstyle`` or ``capstyle``
# parameters.

fig, ax = plt.subplots(figsize=(8, 2))
ax.set_title('Cap style')

for x, style in enumerate(['butt', 'round', 'projecting']):
    ax.text(x+0.25, 1, style, ha='center')
    xx = [x, x+0.5]
    yy = [0, 0]
    ax.plot(xx, yy, lw=12, color='tab:blue', solid_capstyle=style)
    ax.plot(xx, yy, lw=1, color='black')
    ax.plot(xx, yy, 'o', color='tab:red', markersize=3)
ax.text(2.25, 0.7, '(default)', ha='center')

ax.set_ylim(-.5, 1.5)
ax.set_axis_off()


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
matplotlib.axes.Axes.plot
matplotlib.pyplot.plot
