"""
Demo of a basic pie chart plus a few additional features.

In addition to the basic pie chart, this demo shows a few optional features:

    * slice labels
    * auto-labeling the percentage
    * offsetting a slice with "explode"
    * drop-shadow
    * custom start angle

Note about the custom start angle:

The default ``startangle`` is 0, which would start the "Frogs" slice on the
positive x-axis. This example sets ``startangle = 90`` such that everything is
rotated counter-clockwise by 90 degrees, and the frog slice starts on the
positive y-axis.
"""
import numpy as np
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fg1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# Plot four Pie charts in a 2x2 grid:
pie_data = [np.roll(sizes, i) for i in range(4)]  # generate some data
pie_centerpos = [(0, 0), (0, 1), (1, 0), (1, 1)]  # the grid positions

fg2, ax2 = plt.subplots()
for data, cpos in zip(pie_data, pie_centerpos):
    _, txts = ax2.pie(data, explode=explode, shadow=True, startangle=90,
                      radius=0.35, center=cpos, frame=True, labeldistance=.7)
    # Make texts include number and labels:
    for t, l, d in zip(txts, labels, data):
        t.set_text("%s\n %.1f%%" % (l, d))
        t.set_horizontalalignment("center")
        t.set_fontsize(8)

ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Sunny", "Cloudy"])
ax2.set_yticklabels(["Dry", "Rainy"])
ax2.set_xlim((-0.5, 1.5))
ax2.set_ylim((-0.5, 1.5))
ax2.set_aspect('equal')  # Equal aspect ratio ensures that the pie is a circle.

plt.show()
