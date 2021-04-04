"""
==============================
errorbar(X, Y, xerr, yerr,...)
==============================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data:
np.random.seed(1)
X = [2, 4, 6]
Y = [4, 5, 4]
E = np.random.uniform(0.5, 1.5, 3)

# plot:
fig, ax = plt.subplots()

ax.errorbar(X, Y, E, linewidth=2, capsize=6)

ax.set_xlim(0, 8)
ax.set_xticks(np.arange(1, 8))
ax.set_ylim(0, 8)
ax.set_yticks(np.arange(1, 8))

plt.show()
