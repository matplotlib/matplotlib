"""
============
Sigmoid Function
============
A script that showes the sigmoid function with horizontal lines for emphasis of important values.
"""
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 5))
plt.axhline(c="k")
plt.axvline(c="k")
plt.axhline(y=0.5,c="k", ls = ":")
plt.axhline(y=1.0,c="k", ls = ":")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
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
matplotlib.axes.Axes.axhline
matplotlib.axes.Axes.axvline
