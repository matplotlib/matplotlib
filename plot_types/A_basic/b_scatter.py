"""
==================
scatter(X, Y, ...)
==================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('cheatsheet_gallery')

# make the data
np.random.seed(3)
X = 4 + np.random.normal(0, 1.25, 24)
Y = 4 + np.random.normal(0, 1.25, len(X))

# plot
fig, ax = plt.subplots()

ax.scatter(X, Y, 20, zorder=10,
            edgecolor="none", linewidth=0.25)

ax.set_xlim(0, 8)
ax.set_xticks(np.arange(1, 8))
ax.set_ylim(0, 8)
ax.set_yticks(np.arange(1, 8))
plt.show()
