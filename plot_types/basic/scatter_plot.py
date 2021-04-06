"""
==================
scatter(X, Y, ...)
==================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make the data
np.random.seed(3)
X = 4 + np.random.normal(0, 2, 24)
Y = 4 + np.random.normal(0, 2, len(X))
# size and color:
S = np.random.uniform(15, 80, len(X))

# plot
fig, ax = plt.subplots()

ax.scatter(X, Y, s=S, c=-S, cmap=plt.get_cmap('Blues'), vmin=-100, vmax=0)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
