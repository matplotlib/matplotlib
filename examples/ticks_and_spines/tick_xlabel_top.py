"""
==========================================
Set default x-axis tick labels on the top
==========================================

We can use :rc:`xtick.labeltop` (default False) and :rc:`xtick.top`
(default False) and :rc:`xtick.labelbottom` (default True) and
:rc:`xtick.bottom` (default True) to control where on the axes ticks and
their labels appear.
These properties can also be set in the ``.matplotlib/matplotlibrc``.

"""
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

x = np.arange(10)

fig, ax = plt.subplots()

ax.plot(x)
ax.set_title('xlabel top', pad=24)  # increase padding to make room for labels

plt.show()
