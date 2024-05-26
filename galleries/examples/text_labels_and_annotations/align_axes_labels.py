"""
===================================
Axes labels outside floating spines
===================================

Axis labels can be aligned to the spine or the axis box; both are shown here.

"""

import numpy as np
import matplotlib.pyplot as plt

# data generation
x = np.arange(-10, 20, 0.2)
y = 1.0/(1.0+np.exp(-x))


def basic_plot(ax):
    ax.plot(x, y)
    # Limits
    ax.set_ylim(-0.4, 1.1)
    # The labels
    ax.set_xlabel(r'x-axis label')
    ax.set_ylabel(r'y-axis label')

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
fig.suptitle('Axes Labels outside spines')
subfigs = fig.subfigures(nrows=2, ncols=1)

subfigs[0].suptitle('Bottom and left spines', weight='bold')
axs = subfigs[0].subplots(nrows=1, ncols=2)
for ax in axs:
    basic_plot(ax)
    # Eliminate upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Show ticks on the left and lower spines only
    ax.xaxis.set_tick_params(bottom='on', top=False, direction='inout')
    ax.yaxis.set_tick_params(left='on', right=False, direction='inout')
    # Make spines pass through zero of the other spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

ax = axs[0]
ax.set_title('Alongside Spines', {'fontweight': 'bold'})
# Now set label positions
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')

ax = axs[1]
ax.set_title('Outside Axes', {'fontweight': 'bold'})
# Now set label positions
ax.xaxis.set_label_position('axesbottom')
ax.yaxis.set_label_position('axesleft')

# Top and right spines
subfigs[1].suptitle('Top and right spines', weight='bold')
axs = subfigs[1].subplots(nrows=1, ncols=2)
for ax in axs:
    basic_plot(ax)
    # Eliminate bottom and left spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Show ticks on the right and top spines only
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False,
                             bottom=False, top=True, direction='inout')
    ax.yaxis.set_tick_params(labelright=True, labelleft=False,
                             left=False, right=True, direction='inout')
    # Make spines pass through zero of the other axis
    ax.spines['top'].set_position('zero')
    ax.spines['right'].set_position('zero')

ax = axs[0]
ax.set_title('Alongside Spines', {'fontweight': 'bold'})
# Now set label positions
ax.xaxis.set_label_position('top')
ax.yaxis.set_label_position('right')

ax = axs[1]
ax.set_title('Outside Axes', {'fontweight': 'bold'})
# Now set label positions
ax.xaxis.set_label_position('axestop')
ax.yaxis.set_label_position('axesright')

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.subfigures`
#    - `matplotlib.figure.Figure.subplots`
#    - `matplotlib.axis.Axis.set_label_position`
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_ylabel`
#    - `matplotlib.axes.Axes.set_ylim`
