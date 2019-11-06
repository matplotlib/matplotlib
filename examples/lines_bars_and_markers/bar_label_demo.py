"""
==============
Bar Label Demo
==============

This example shows how to use `bar_label` helper function
to create bar chart labels.

See also the :doc:`grouped bar
</gallery/lines_bars_and_markers/barchart>`,
:doc:`stacked bar
</gallery/lines_bars_and_markers/bar_stacked>` and
:doc:`horizontal bar chart
</gallery/lines_bars_and_markers/barh>` examples.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Define the data

N = 5
menMeans = (20, 35, 30, 35, -27)
womenMeans = (25, 32, 34, 20, -25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

###############################################################################
# Grouped bar chart

fig1, ax1 = plt.subplots()

rects1 = ax1.bar(ind - width/2, menMeans, width, label='Men')
rects2 = ax1.bar(ind + width/2, womenMeans, width, label='Women')

ax1.set_ylabel('Scores')
ax1.set_title('Scores by group and gender')
ax1.set_xticks(ind)
ax1.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax1.legend()

# Basic labels
ax1.bar_label(rects1)
ax1.bar_label(rects2)

plt.show()

###############################################################################
# Stacked bar plot with error bars

fig2, ax2 = plt.subplots()

p1 = ax2.bar(ind, menMeans, width, yerr=menStd, label='Men')
p2 = ax2.bar(ind, womenMeans, width,
             bottom=menMeans, yerr=womenStd, label='Women')

ax2.set_ylabel('Scores')
ax2.set_title('Scores by group and gender')
ax2.set_xticks(ind)
ax2.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax2.legend()

# Label with 'center' mode instead of the default 'edge' mode
ax2.bar_label(p1, mode='center')
ax2.bar_label(p2, mode='center')
ax2.bar_label(p2)

plt.show()

###############################################################################
# Horizontal bar chart

# Fixing random state for reproducibility
np.random.seed(19680801)

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

fig3, ax3 = plt.subplots()

hbars1 = ax3.barh(y_pos, performance, xerr=error, align='center')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(people)
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.set_xlabel('Performance')
ax3.set_title('How fast do you want to go today?')

# Label with specially formatted floats
ax3.bar_label(hbars1, fmt='%.2f')

plt.show()

###############################################################################
# Some of the more advanced things that one can do with bar labels

fig4, ax4 = plt.subplots()

hbars2 = ax4.barh(y_pos, performance, xerr=error, align='center')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(people)
ax4.invert_yaxis()  # labels read top-to-bottom
ax4.set_xlabel('Performance')
ax4.set_title('How fast do you want to go today?')

# Label with given captions, custom padding, shifting and annotate option
arrowprops = dict(color='b', arrowstyle="-|>",
                  connectionstyle="angle,angleA=0,angleB=90,rad=20")

ax4.bar_label(hbars2, captions=['±%.2f' % e for e in error],
              padding=30, shifting=20, arrowprops=arrowprops, color='b')

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods and classes is shown
# in this example:

matplotlib.axes.Axes.bar
matplotlib.pyplot.bar
matplotlib.axes.Axes.barh
matplotlib.pyplot.barh
matplotlib.axes.Axes.bar_label
matplotlib.pyplot.bar_label
