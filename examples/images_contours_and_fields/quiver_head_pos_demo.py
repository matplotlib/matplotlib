"""
==================
Quiver Arrowhead Position Demo
==================

An example of how to add positioning of arrowheads to
a `~.axes.Axes.quiver` plot with a `~.axes.Axes.quiverkey`.

For more advanced options refer to
:doc:`/gallery/images_contours_and_fields/quiver_demo`.
"""
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-10, 10, 2)
Y = np.arange(-10, 10, 2)
U, V = np.meshgrid(X, Y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, head_pos='mid', mid_scale=0.6)
ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions and methods is shown in this example:

import matplotlib
matplotlib.axes.Axes.quiver
matplotlib.pyplot.quiver
matplotlib.axes.Axes.quiverkey
matplotlib.pyplot.quiverkey