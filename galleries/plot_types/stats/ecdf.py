"""
=======
ecdf(x)
=======
Compute and plot the empirical cumulative distribution function of x.

See `~matplotlib.axes.Axes.ecdf`.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

# plot:
fig, ax = plt.subplots()
ax.ecdf(x)
plt.show()
