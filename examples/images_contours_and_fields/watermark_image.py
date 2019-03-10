"""
===============
Watermark image
===============

Using a PNG file as a watermark.
"""

import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt


with cbook.get_sample_data('logo2.png') as file:
    im = image.imread(file)

fig, ax = plt.subplots()

ax.plot(np.sin(10 * np.linspace(0, 1)), '-o', ms=20, alpha=0.7, mfc='orange')
ax.grid()
fig.figimage(im, 10, 10, zorder=3, alpha=.5)

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
matplotlib.image
matplotlib.image.imread
matplotlib.pyplot.imread
matplotlib.figure.Figure.figimage
