"""
========================
hist(X, [bins],...)
========================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data
np.random.seed(1)
X = 4 + np.random.normal(0, 1.5, 200)

# plot:
fig, ax = plt.subplots()

ax.hist(X, bins=8, linewidth=0.5, edgecolor="white")

ax.set_xlim(0, 8)
ax.set_xticks(np.arange(1, 8))
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(1, 80, 10))

plt.show()
