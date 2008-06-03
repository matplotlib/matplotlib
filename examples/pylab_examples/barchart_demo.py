
#!/usr/bin/env python
# a bar plot with errorbars
from numpy import arange
from matplotlib.pyplot import *

N = 5
menMeans = (20, 35, 30, 35, 27)
menStd =   (2, 3, 4, 1, 2)

ind = arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

figure()
subplot(111)
rects1 = bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd =   (3, 5, 2, 3, 3)
rects2 = bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

# add some
ylabel('Scores')
title('Scores by group and gender')
xticks(ind+width, ('G1', 'G2', 'G3', 'G4', 'G5') )

legend( (rects1[0], rects2[0]), ('Men', 'Women') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
#savefig('barchart_demo')
show()
