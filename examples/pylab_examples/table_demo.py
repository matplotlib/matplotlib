#!/usr/bin/env python
import matplotlib

from pylab import *
from colours import get_colours

axes([0.2, 0.2, 0.7, 0.6])   # leave room below the axes for the table

data = [[  66386,  174296,   75131,  577908,   32015],
        [  58230,  381139,   78045,   99308,  160454],
        [  89135,   80552,  152558,  497981,  603535],
        [  78415,   81858,  150656,  193263,   69638],
        [ 139361,  331509,  343164,  781380,   52269]]

colLabels = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rowLabels = ['%d year' % x for x in (100, 50, 20, 10, 5)]

# Get some pastel shades for the colours
colours = get_colours(len(colLabels))
colours.reverse()
rows = len(data)

ind = arange(len(colLabels)) + 0.3  # the x locations for the groups
cellText = []
width = 0.4     # the width of the bars
yoff = array([0.0] * len(colLabels)) # the bottom values for stacked bar chart
for row in xrange(rows):
    bar(ind, data[row], width, bottom=yoff, color=colours[row])
    yoff = yoff + data[row]
    cellText.append(['%1.1f' % (x/1000.0) for x in yoff])

# Add a table at the bottom of the axes
colours.reverse()
cellText.reverse()
the_table = table(cellText=cellText,
                  rowLabels=rowLabels, rowColours=colours,
                  colLabels=colLabels,
                  loc='bottom')
ylabel("Loss $1000's")
vals = arange(0, 2500, 500)
yticks(vals*1000, ['%d' % val for val in vals])
xticks([])
title('Loss by Disaster')
#savefig('table_demo_small', dpi=75)
#savefig('table_demo_large', dpi=300)

show()
