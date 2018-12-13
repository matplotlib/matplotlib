"""
=====
Fig X
=====

Add lines to a figure (without axes).
"""
import matplotlib.pyplot as plt
import matplotlib.lines as lines


fig = plt.figure()

l1 = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)

l2 = lines.Line2D([0, 1], [1, 0], transform=fig.transFigure, figure=fig)

fig.lines.extend([l1, l2])

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
