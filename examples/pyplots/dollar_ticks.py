"""
============
Dollar Ticks
============

Use a `~.ticker.FormatStrFormatter` to prepend dollar signs on y axis labels.
"""
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.subplots()
ax.plot(100*np.random.rand(20))

# Use automatic StrMethodFormatter
ax.yaxis.set_major_formatter('${x:1.2f}')

ax.yaxis.set_tick_params(which='major', labelcolor='green',
                         labelleft=False, labelright=True)

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
matplotlib.pyplot.subplots
matplotlib.axis.Axis.set_major_formatter
matplotlib.axis.Axis.set_tick_params
matplotlib.axis.Tick
matplotlib.ticker.StrMethodFormatter
