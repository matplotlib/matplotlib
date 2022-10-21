"""
=============
Pyplot Simple
=============

A very simple pyplot where a list of numbers are plotted against their
index. Creates a straight line due to the rate of change being 1 for
both the X and Y axis. Use a format string (here, 'o-r') to set the
markers (circles), linestyle (solid line) and color (red).

.. redirect-from:: /gallery/pyplots/fig_axes_labels_simple
.. redirect-from:: /gallery/pyplots/pyplot_formatstr
"""
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], 'o-r')
plt.ylabel('some numbers')
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.plot`
#    - `matplotlib.pyplot.ylabel`
#    - `matplotlib.pyplot.show`
