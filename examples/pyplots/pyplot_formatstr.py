"""
================
Pyplot Formatstr
================

Use a format string to colorize a `~matplotlib.axes.Axes.plot` and set its
markers.
"""
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
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
matplotlib.pyplot.plot
matplotlib.axes.Axes.plot
