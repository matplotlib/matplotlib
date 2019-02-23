"""
========================
Colorbar Annotation Demo
========================

Demonstrates how one can annotate the colorbar without needing to also add
lines to a contour plot.

Note that this is not a typical use case. The more common use case for
annotations is to add lines to both a contour plot *and* the corresponding
location on the colorbar. In this more common use case,
`~matplotlib.colorbar.Colorbar.add_lines` can be used.
"""

import matplotlib.pyplot as plt
import numpy as np

# Create sample values to plot
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-X**2 - Y**2)

# Choose a value to annotate
desired_cbar_yvalue = 0.90

# Build filled contour plot
surfplot = plt.contourf(X, Y, Z, cmap=plt.cm.binary_r)
cbar = plt.colorbar(surfplot)

# Find min and max values on colorbar
cmin = np.float(cbar.ax.get_yticklabels()[0].get_text())
cmax = np.float(cbar.ax.get_yticklabels()[-1].get_text())

# Use min/max values to calculate annotation location
loc = (desired_cbar_yvalue - cmin) / (cmax - cmin)

# Annotate the colorbar
cbar.ax.axhline(loc, color='r', linewidth=5)

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods and classes is shown
# in this example:

import matplotlib
matplotlib.colorbar

