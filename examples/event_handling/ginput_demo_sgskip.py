"""
===========
Ginput Demo
===========

This provides examples of uses of interactive functions, such as ginput,

"""


import matplotlib.pyplot as plt
import numpy as np
t = np.arange(10)
plt.plot(t, np.sin(t))
print("Please click")
x = plt.ginput(3)
print("clicked", x)
plt.show()
