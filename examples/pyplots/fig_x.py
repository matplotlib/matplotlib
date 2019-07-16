"""
=======================
Adding lines to figures
=======================

Adding lines to a figure without any axes.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as lines


fig = plt.figure()
fig.add_artist(lines.Line2D([0, 1], [0, 1]))
fig.add_artist(lines.Line2D([0, 1], [1, 0]))
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
matplotlib.pyplot.figure
matplotlib.lines
matplotlib.lines.Line2D
