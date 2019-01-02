"""
==========================
Figure legend outside axes
==========================

Instead of plotting a legend on each axis, a legend for all the artists on all
the sub-axes of a figure can be plotted instead.  If constrained layout is
used (:doc:`/tutorials/intermediate/constrainedlayout_guide`) then room
can be made automatically for the legend by using `~.Figure.legend` with the
``outside=True`` kwarg.

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

fig.legend(loc='upper center', outside=True, ncol=2)
fig.legend(ax=[axs[1]], outside=True, loc='lower right')
fig.legend(handles=[h2, h4], labels=['curve2', 'curve4'],
                   outside=True, loc='center left', borderaxespad=6)
plt.show()

###############################################################################
# The usual codes for the *loc* kwarg are allowed, however, the corner
# codes have an ambiguity as to whether the legend is stacked
# horizontally (the default) or vertically.  To specify the vertical stacking
# the *outside* kwarg can be specified with ``"vertical"`` instead of just
# the booloean *True*:

fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)
axs[0].plot(x, y1, 'rs-', label='Line1')
h2, = axs[0].plot(x, y2, 'go', label='Line2')

axs[0].set_ylabel('DATA')
axs[1].plot(x, y3, 'yd-', label='Line3')
h4, = axs[1].plot(x, y4, 'k^', label='Line4')

fig.legend(loc='upper right', outside='vertical', ncol=2)
plt.show()

###############################################################################
# Significantly more complicated layouts are possible using the gridspec
# organization of subplots:

fig = plt.figure(constrained_layout=True)
gs0 = fig.add_gridspec(1, 2)

gs = gs0[0].subgridspec(1, 1)
for i in range(1):
    ax = fig.add_subplot(gs[i,0])
    ax.plot(range(10), label=f'Boo{i}')
lg = fig.legend(ax=[ax], loc='upper left', outside=True, borderaxespad=4)

gs2 = gs0[1].subgridspec(3, 1)
axx = []
for i in range(3):
    ax = fig.add_subplot(gs2[i, 0])
    ax.plot(range(10), label=f'Who{i}', color=f'C{i+1}')
    if i < 2:
        ax.set_xticklabels('')
    axx += [ax]
lg2 = fig.legend(ax=axx[:-1], loc='upper right', outside=True, borderaxespad=4)
plt.show()
