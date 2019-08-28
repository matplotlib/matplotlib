"""
=========
Step Demo
=========

This example demonstrates the use of `.pyplot.step` for piece-wise constant
curves. In particular, it illustrates the effect of the parameter *where*
on the step position.

The circular markers created with `.pyplot.plot` show the actual data
positions so that it's easier to see the effect of *where*.

"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(14)
y = np.sin(x / 2)

plt.step(x, y + 2, label='pre (default)')
plt.plot(x, y + 2, 'C0o', alpha=0.5)

plt.step(x, y + 1, where='mid', label='mid')
plt.plot(x, y + 1, 'C1o', alpha=0.5)

plt.step(x, y, where='post', label='post')
plt.plot(x, y, 'C2o', alpha=0.5)

plt.legend(title='Parameter where:')
plt.show()

# Plotting with where='between'/'edges'
values = np.array([6, 14, 32, 37, 48, 32, 21,  4])  # hist
edges = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])  # bins
fig, axes = plt.subplots(3, 2)
axes = axes.flatten()
axes[0].step(edges, values, where='between')
axes[1].step(values, edges, where='between')
axes[2].step(edges, values, where='edges')
axes[3].step(values, edges, where='edges')
axes[4].step(edges, values, where='edges')
axes[4].semilogy()
axes[5].step(edges, values, where='edges')
axes[5].semilogy()

fig.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.axes.Axes.step
matplotlib.pyplot.step
