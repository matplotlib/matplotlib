"""
========================
What's New 0.98.4 Legend
========================

Create a legend and tweak it with a shadow and a box.
"""
import matplotlib.pyplot as plt
import numpy as np


ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
for n in [1, 2, 3, 4]:
    plt.plot(t1, t1**n, label=f"n={n}")

leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)


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
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
matplotlib.legend.Legend
matplotlib.legend.Legend.get_frame
