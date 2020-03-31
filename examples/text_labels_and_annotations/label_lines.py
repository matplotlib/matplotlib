"""
====================================
Line labels using pre-defined labels
====================================

Defining line labels with plots.
"""


import numpy as np
import matplotlib.pyplot as plt

# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c + d, 'k', label='Total message length')

ax.label_lines()

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

#import matplotlib
#matplotlib.axes.Axes.plot
#matplotlib.pyplot.plot
#matplotlib.axes.Axes.label_lines
#matplotlib.pyplot.label_lines
