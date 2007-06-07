#!/usr/bin/env python
"""
Make a pie chart - see
http://matplotlib.sf.net/matplotlib.pylab.html#-pie for the docstring.

This example shows a basic pie chart with labels in figure 1, and
figure 2 uses a couple of optional features, like autolabeling the
area percentage, offsetting a slice using "explode" and addind a shadow
for a 3D effect
"""
from pylab import *

# make a square figure and axes
figure(1, figsize=(8,8))
ax = axes([0.1, 0.1, 0.8, 0.8])

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15,30,45, 10]

figure(1)
pie(fracs, labels=labels)

# figure(2) showa some optional features.  autopct is used to label
# the percentage of the pie, and can be a format string or a function
# which takes a percentage and returns a string.  explode is a
# len(fracs) sequuence which gives the fraction of the radius to
# offset that slice.

figure(2, figsize=(8,8))
explode=(0, 0.05, 0, 0)
pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

savefig('pie_demo')
show()

