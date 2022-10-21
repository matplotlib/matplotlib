"""
=============
scatter(x, y)
=============

See `~matplotlib.axes.Axes.scatter`.
"""
import matplotlib.pyplot as plt
import numpy as np



# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))
# size and color:
sizes = 5*np.random.uniform(15, 80, len(x))

colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes,alpha=0.5, cmap='nipy_spectral', c=colors, vmin=0, vmax=100,marker='+')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')


plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()
