"""
=================================
fill[_between][x](X, Y1, Y2, ...)
=================================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('cheatsheet_gallery')

# make data
np.random.seed(1)
X = np.linspace(0, 8, 16)
Y1 = 3 + 4*X/8 + np.random.uniform(0.0, 0.5, len(X))
Y2 = 1 + 2*X/8 + np.random.uniform(0.0, 0.5, len(X))

# plot
fig, ax = plt.subplots()

ax.fill_between(X, Y1, Y2, alpha=.5, linewidth=0)
ax.plot(X, (Y1+Y2)/2, linewidth=2.5)

ax.set_xlim(0, 8)
ax.set_xticks(np.arange(1, 8))
ax.set_ylim(0, 8)
ax.set_yticks(np.arange(1, 8))

plt.show()
