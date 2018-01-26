"""
===========
Interp Demo
===========

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import stineman_interp

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
yp = None
xi = np.linspace(x[0], x[-1], 100)
yi = stineman_interp(xi, x, y, yp)

fig, ax = plt.subplots()
ax.plot(x, y, 'o', xi, yi, '.')
plt.show()
