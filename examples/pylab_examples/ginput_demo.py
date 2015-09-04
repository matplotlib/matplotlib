# -*- noplot -*-

from __future__ import print_function
# from pylab import arange, plot, sin, ginput, show
import matplotlib.pyplot as plt
import numpy as np
from pylab import ginput

t = np.arange(10)
plt.plot(t, np.sin(t))
print("Please click")
x = ginput(3)
print("clicked", x)
plt.show()
