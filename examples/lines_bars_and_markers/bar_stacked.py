"""
=================
Stacked bar chart
=================

This is an example of creating a stacked bar plot with error bars
using `~matplotlib.pyplot.bar`.  Note the parameters *yerr* used for
error bars, and *bottom* to stack the coffee's bars on top of the tea's
bars.
"""

import matplotlib.pyplot as plt


labels = ['G1', 'G2', 'G3', 'G4', 'G5']
tea_means = [20, 35, 30, 35, 27]
coffee_means = [25, 32, 34, 20, 25]
tea_std = [2, 3, 4, 1, 2]
coffee_std = [3, 5, 5, 3, 3]
width = 0.25       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, tea_means, width, yerr=tea_std, label='Tea')
ax.bar(labels, coffee_means, width, yerr=coffee_std, bottom=tea_means,
       label='Coffee')

ax.set_ylabel('Scores')
ax.set_title('Scores by group and beverage preferences')
ax.legend()

plt.show()
