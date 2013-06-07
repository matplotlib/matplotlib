"""
Voronoi diagrams of unstructured grids. Optionally pseudocolored.
"""

import matplotlib.pyplot as plt
import numpy as np
import math


# Some data to work with
N = 128
r = np.random.random(N) * 3 + 0.5
phi = np.random.random(N) * 2 * math.pi
T = r ** 0.5

# Plot pseudocolored Voronoi diagram
plt.figure()
plt.gca().set_aspect('equal')
plt.voronoi(r * np.sin(phi), r * np.cos(phi), T, cmap='Spectral')
plt.colorbar()
plt.axis((-3.5, 3.5, -3.5, 3.5))
plt.title('Voronoi diagram, pseudocolored')

# Plot Voronoi diagram with cell-generating points
plt.figure()
plt.gca().set_aspect('equal')
plt.voronoi(r * np.sin(phi), r * np.cos(phi), facecolors='1.0', edgecolor='0.5', lw=2.0)
plt.scatter(r * np.sin(phi), r * np.cos(phi), s=8, c='red', linewidths=0.0)
plt.axis((-3.5, 3.5, -3.5, 3.5))
plt.title('Voronoi diagram with cell-generating points')

# Show it
plt.show()

