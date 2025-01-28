"""
==============
stairs(values)
==============
Draw a stepwise constant function as a line or a filled plot.

See `~matplotlib.axes.Axes.stairs` when plotting :math:`y` between
:math:`(x_i, x_{i+1})`. For plotting :math:`y` at :math:`x`, see
`~matplotlib.axes.Axes.step`.

.. redirect-from:: /plot_types/basic/step
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

# plot
fig, ax = plt.subplots()

ax.stairs(y, linewidth=2.5)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
