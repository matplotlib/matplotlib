"""
====================
Pyplot striped line
====================

Plot a striped line plots in a single call to `~matplotlib.pyplot.plot`, and a correct legend from `~matplotlib.pyplot.legend`.

"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1., 3., 10)
y = x**3

plt.plot(x, y, linestyle = '--', color = 'orange', offcolor = 'blue',  label = 'a stripped line')
plt.legend()
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
matplotlib.pyplot.plot
matplotlib.pyplot.show
matplotlib.pyplot.legend

