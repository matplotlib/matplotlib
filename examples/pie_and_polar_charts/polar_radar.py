#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
pydocstyle --ignore=D101,D213
==========
Radar plot
==========

The following example shows a way to plot Radar chart in matplotlib.
Such charts are a way to visualize multivariate data.
They are used to plot one or more groups of values over
multiple common variables.

Here, school subjects are common variables.
Group of values are the scores obtained in those
subjects by two students.

"""

import matplotlib.pyplot as plt
import pandas as pd
from math import pi

values = [  # exam scores out of 100 for Student 1
    60,
    70,
    67,
    78,
    83,
    94,
    58,
    ]
values2 = [  # exam scores out of 100 for Student 2
    90,
    88,
    78,
    99,
    82,
    84,
    60,
    ]
subjects = [
    'Second Language',
    'Math',
    'Music',
    'History',
    'Science',
    'Sport',
    'Art',
    ]
N = len(values)
values += values[:1]
values2 += values2[:1]

ax = plt.subplot(111, polar=True)

# angles for each category

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax.plot(angles, values, linewidth=2, linestyle='solid',
        label='Student 1')
ax.plot(angles, values2, linewidth=2, linestyle='solid',
        label='Student 2')

# plotting

plt.xticks(angles[:-1], subjects)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

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
matplotlib.pyplot
matplotlib.axes.Axes.legend
matplotlib.axes.Axes.plot
matplotlib.projections.polar
