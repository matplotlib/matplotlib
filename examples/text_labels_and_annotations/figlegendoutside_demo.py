"""
==========================
Figure legend outside axes
==========================

Instead of plotting a legend on each axis, a legend for all the artists on all
the sub-axes of a figure can be plotted instead.  If constrained layout is
used (:doc:`/tutorials/intermediate/constrainedlayout_guide`) then room
can be made automatically for the legend by using `~.Figure.legend_outside`.

"""

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)
axs[0].plot(x, y1, 'rs-', label='Line1')
h2, = axs[0].plot(x, y2, 'go', label='Line2')

axs[0].set_ylabel('DATA')

y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)
axs[1].plot(x, y3, 'yd-', label='Line3')
h4, = axs[1].plot(x, y4, 'k^', label='Line4')

fig.legend_outside(loc='upper center', ncol=2)
fig.legend_outside(axs=[axs[1]], loc='lower right')
fig.legend_outside(handles=[h2, h4], labels=['curve2', 'curve4'],
                   loc='center left', borderaxespad=6)
plt.show()
