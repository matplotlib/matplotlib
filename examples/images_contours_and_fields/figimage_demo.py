"""
=============
Figimage Demo
=============

This illustrates placing images directly in the figure, with no Axes objects.

"""
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
Z = np.arange(10000).reshape((100, 100))
Z[:, 50:] = 1

im1 = fig.figimage(Z, xo=50, yo=0, origin='lower')
im2 = fig.figimage(Z, xo=100, yo=100, alpha=.8, origin='lower')

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure`
#    - `matplotlib.figure.Figure.figimage` / `matplotlib.pyplot.figimage`
