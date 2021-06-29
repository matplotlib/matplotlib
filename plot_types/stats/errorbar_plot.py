"""
==========================
errorbar(x, y, yerr, xerr)
==========================

See `~matplotlib.axes.Axes.errorbar`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data:
np.random.seed(1)
x = [2, 4, 6]
y = [4, 5, 4]
yerr = np.random.uniform(0.5, 1.5, 3)

# plot:
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr, linewidth=2, capsize=6)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
