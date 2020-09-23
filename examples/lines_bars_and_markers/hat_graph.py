"""
===============================
Hat Graph with labels
===============================
This example shows a how to create a hat graph
and how to annotate with labels.
Refer (https://doi.org/10.1186/s41235-019-0182-3)
to know more about hat graph
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# initialise labels and a numpy array make sure you have
# N labels of N number of values in the array
labels = ['I', 'II', 'III', 'IV', 'V']
playerA = np.array([5, 15, 22, 20, 25])
playerB = np.array([25, 32, 34, 30, 27])
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, np.zeros_like(playerA), width,
                bottom=playerA, label='Player A', fill=False)
rects2 = ax.bar(x + width/2, playerB - playerA, width,
                bottom=playerA, label='Player B', edgecolor='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0, 60)
ax.set_ylabel('Score')
ax.set_title('Scores by number of game and Players')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_xlabel('Games')


def Label(heights, rects):
    """Attach a text label on top of each bar."""
    i = 0
    for rect in rects:
        height = int(heights[i])
        i += 1
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset.
                    textcoords="offset points",
                    ha='center', va='bottom')
Label(playerA, rects1)
Label(playerB, rects2)
fig.tight_layout()
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
matplotlib.axes.Axes.annotate
matplotlib.pyplot.annotate
