#!/usr/bin/env python
"""
You can control the axis tick and grid properties
"""

from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s)
grid(True)

# matlab handle graphics style
xticklines = get(gca(), 'xticklines')
yticklines = get(gca(), 'yticklines')
xgridlines = get(gca(), 'xgridlines')
ygridlines = get(gca(), 'ygridlines')
xticklabels = get(gca(), 'xticklabels')
yticklabels = get(gca(), 'yticklabels')

setp(xticklines, 'linewidth', 3)
setp(yticklines, 'linewidth', 3)
setp(xgridlines, 'linestyle', '-')
setp(ygridlines, 'linestyle', '-')
setp(yticklabels, 'color', 'r', fontsize='medium')
setp(xticklabels, 'color', 'r', fontsize='medium')

# keyword args are legal too
#set(xticklabels, color='r', fontsize='medium')

#savefig('axprops_demo')
show()


"""
# the same script, python style
from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
ax = subplot(111)
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

#savefig('axprops_demo')
show()

"""
