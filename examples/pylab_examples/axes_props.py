#!/usr/bin/env python
"""
You can control the axis tick and grid properties
"""

from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s)
grid(True)

# MATLAB style
xticklines = getp(gca(), 'xticklines')
yticklines = getp(gca(), 'yticklines')
xgridlines = getp(gca(), 'xgridlines')
ygridlines = getp(gca(), 'ygridlines')
xticklabels = getp(gca(), 'xticklabels')
yticklabels = getp(gca(), 'yticklabels')

setp(xticklines, 'linewidth', 3)
setp(yticklines, 'linewidth', 3)
setp(xgridlines, 'linestyle', '-')
setp(ygridlines, 'linestyle', '-')
setp(yticklabels, 'color', 'r', fontsize='medium')
setp(xticklabels, 'color', 'r', fontsize='medium')



show()


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
