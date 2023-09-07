"""
=============
scatter(x, y)
=============

See `~matplotlib.axes.Axes.scatter`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data
rng = np.random.default_rng(seed=None)

x = 4 + rng.normal(0, 2, 24)
y = 4 + rng.normal(0, 2, len(x))
# size and color:
sizes = rng.uniform(15, 80, len(x))
colors = rng.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
