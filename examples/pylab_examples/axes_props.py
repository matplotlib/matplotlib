#!/usr/bin/env python
"""
You can control the axis tick and grid properties
"""

import matplotlib.pyplot as plt
import numpy as np 

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*pi*t)
plt.plot(t, s)
plt.grid(True)

# MATLAB style
xticklines = plt.getp(plt.gca(), 'xticklines')
yticklines = plt.getp(plt.gca(), 'yticklines')
xgridlines = plt.getp(plt.gca(), 'xgridlines')
ygridlines = plt.getp(plt.gca(), 'ygridlines')
xticklabels = plt.getp(plt.gca(), 'xticklabels')
yticklabels = plt.getp(plt.gca(), 'yticklabels')

plt.setp(xticklines, 'linewidth', 3)
plt.setp(yticklines, 'linewidth', 3)
plt.setp(xgridlines, 'linestyle', '-')
plt.setp(ygridlines, 'linestyle', '-')
plt.setp(yticklabels, 'color', 'r', fontsize='medium')
plt.setp(xticklabels, 'color', 'r', fontsize='medium')


plt.show()


"""
# the same script, python style
from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
fig, ax = plt.subplots()
ax.plot(t, s)
ax.grid(True)

ticklines = ax.get_xticklines()
ticklines.extend( ax.get_yticklines() )
gridlines = ax.get_xgridlines()
gridlines.extend( ax.get_ygridlines() )
ticklabels = ax.get_xticklabels()
ticklabels.extend( ax.get_yticklabels() )

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('r')
    label.set_fontsize('medium')

show()

"""
