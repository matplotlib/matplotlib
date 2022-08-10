"""
=================================
Error bar rendering on polar axis
=================================

Demo of error bar plot in polar coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
theta = np.arange(0, 2*np.pi, np.pi / 4)
r = theta / np.pi / 2 + 0.5
ax.errorbar(theta, r, xerr=0.25, yerr=0.1, capsize=7, fmt="o")
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
#    - `matplotlib.projections.polar`
