"""
=============================
The object-oriented interface
=============================

A pure object-oriented example using the agg backend. Notice that there is no
``pyplot`` used here.
"""

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

fig = Figure()
# A canvas must be manually attached to the figure (pyplot would automatically
# do it).  This is done by instantiating the canvas with the figure as
# argument.
FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.plot([1, 2, 3])
ax.set_title('hi mom')
ax.grid(True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
fig.savefig('test')

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
matplotlib.backends.backend_agg.FigureCanvasAgg
matplotlib.figure.Figure
matplotlib.figure.Figure.add_subplot
matplotlib.figure.Figure.savefig
matplotlib.axes.Axes.plot
matplotlib.axes.Axes.set_title
matplotlib.axes.Axes.grid
matplotlib.axes.Axes.set_xlabel
matplotlib.axes.Axes.set_ylabel
