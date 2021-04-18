"""
================================
Cross- and Auto-Correlation Demo
================================

Example use of cross-correlation (`~.Axes.xcorr`) and auto-correlation
(`~.Axes.acorr`) plots.
"""
import matplotlib.pyplot as plt
import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)


x, y = np.random.randn(2, 100)
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
ax1.grid(True)

ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
ax2.grid(True)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.acorr` / `matplotlib.pyplot.acorr`
#    - `matplotlib.axes.Axes.xcorr` / `matplotlib.pyplot.xcorr`
