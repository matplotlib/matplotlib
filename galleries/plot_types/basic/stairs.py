"""
==============
stairs(x, y)
==============

See `~matplotlib.axes.Axes.stairs`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
x = np.arange(14)
centers = x[:-1] + np.diff(x) / 2
y = -(centers - 4) ** 2 + 8

# plot
fig, ax = plt.subplots()

ax.stairs(y - 1, x, baseline=None, label='stairs()')
plt.plot(centers, y - 1, 'o--', color='grey', alpha=0.3)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
