"""
======================
bar[h](x, height, ...)
======================
"""
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('mpl_plot_gallery')

# make data:
np.random.seed(3)
X = 0.5 + np.arange(8)
Y = np.random.uniform(2, 7, len(X))

# plot
fig, ax = plt.subplots()

ax.bar(X, Y, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
