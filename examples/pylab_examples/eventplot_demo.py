#!/usr/bin/env python
# -*- Coding:utf-8 -*-
# an eventplot showing sequences of events with various line properties
# the plot is shown in both horizontal and vertical orientations

import matplotlib.pyplot as plt
import numpy as np

# create random data
np.random.seed(0)
data = np.random.random([6, 50])

# set different colors for each set of positions
colors = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 1, 0],
                   [1, 0, 1],
                   [0, 1, 1]])

# set different line properties for each set of positions
# note that some overlap
lineoffsets = np.array([-15, -3, 1, 1.5, 6, 10])
linelengths = [5, 2, 1, 1, 3, 1.5]

fig = plt.figure()

# create a horizontal plot
ax1 = fig.add_subplot(211)
ax1.eventplot(data, colors=colors, lineoffsets=lineoffsets,
             linelengths=linelengths)
ax1.set_title('horizontal eventplot')


# create a vertical plot
ax2 = fig.add_subplot(212)
ax2.eventplot(data, colors=colors, lineoffsets=lineoffsets,
             linelengths=linelengths, orientation='vertical')
ax2.set_title('vertical eventplot')

fig.show()
