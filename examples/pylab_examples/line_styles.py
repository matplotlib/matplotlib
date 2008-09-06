#!/usr/bin/env python
# This should probably be replaced with a demo that shows all
# line and marker types in a single panel, with labels.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

t = np.arange(0.0, 1.0, 0.1)
s = np.sin(2*np.pi*t)
linestyles = ['_', '-', '--', ':']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = linestyles + markers

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')


axisNum = 0
for row in range(5):
    for col in range(5):
        axisNum += 1
        ax = plt.subplot(5, 5, axisNum)
        style = styles[axisNum % len(styles) ]
        color = colors[axisNum % len(colors) ]
        plt.plot(t,s, style + color, markersize=10)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

plt.show()
