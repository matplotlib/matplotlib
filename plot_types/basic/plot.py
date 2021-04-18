"""
======================
plot([X], Y, [fmt]...)
======================
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data
X = np.linspace(0, 10, 100)
Y = 4 + 2 * np.sin(2 * X)

# plot
fig, ax = plt.subplots()

ax.plot(X, Y, linewidth=2.0)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
