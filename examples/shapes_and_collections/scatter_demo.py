"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt

def scatter_demo(ax, N=50, max_radius=15, alpha=0.5):
	N = N
	x = np.random.rand(N)
	y = np.random.rand(N)
	colors = np.random.rand(N)
	area = np.pi * (max_radius * np.random.rand(N))**2 #0 to max_radius point radiuses

	c = ax.scatter(x, y, s=area, c=colors, alpha=alpha)
	return c

ax = plt.subplot(111)
scatter_demo(ax)
plt.show()


