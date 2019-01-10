"""
===============
Errorbar Limits
===============

Illustration of upper and lower limit symbols on errorbars.
"""

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
x = np.arange(10)
y = np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)

plt.errorbar(x, y, yerr=yerr)

plt.errorbar(x, y + 1, yerr=yerr, uplims=True)

plt.errorbar(x, y + 2, yerr=yerr, uplims=True, lolims=True)

upperlimits = [True, False] * 5
lowerlimits = [False, True] * 5
plt.errorbar(x, y + 3, yerr=yerr, uplims=upperlimits, lolims=lowerlimits)

###############################################################################

fig = plt.figure()
x = np.arange(10) / 10
y = (x + 0.1)**2

plt.errorbar(x, y, xerr=0.1, xlolims=True)
y = (x + 0.1)**3

plt.errorbar(x + 0.6, y, xerr=0.1, xuplims=upperlimits, xlolims=lowerlimits)

y = (x + 0.1)**4
plt.errorbar(x + 1.2, y, xerr=0.1, xuplims=True)

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
matplotlib.axes.Axes.errorbar
matplotlib.pyplot.errorbar
