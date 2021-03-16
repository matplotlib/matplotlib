"""
==================================
hist2d(x, y, [(xbins, ybins)],...)
==================================
"""
import matplotlib.pyplot as plt
import numpy as np

# make data: correlated + noise
np.random.seed(1)
x = np.random.randn(5000)
y = 1.2 * x + np.random.randn(5000)/3

# plot:
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    ax.hist2d(x, y, bins=(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1)))

ax.set_xlim(-2, 2)
ax.set_ylim(-3, 3)

plt.show()
