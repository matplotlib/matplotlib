"""
===========
ginput demo
===========

"""

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(10)
plt.plot(t, np.sin(t))
print("Please click at three points.")
x = plt.ginput(3)
print("clicked", x)
plt.show()
