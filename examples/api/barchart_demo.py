#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import sys

#----------------------   
def barchart(ax, yLabel, xLabel, fig_title, bar_width, lst_bar_heights, lst_stdevs, lst_bar_colors, lst_legends=[]):
                
    nSeries = len(lst_bar_heights)
    nBins   = len(lst_bar_heights[0]) 
    ind     = np.arange(nBins)  # the x locations for the groups
    width   = bar_width       # the width of the bars
    
    lst_rects = [ax.bar(ind+n*width, lst_bar_heights[n], width, color=lst_bar_colors[n], yerr=lst_stdevs[n]) for n in range(nSeries)]
    ax.set_xticklabels(xLabel)
    
    if lst_legends:
        ax.legend( lst_rects, [legend_i for legend_i in lst_legends] )
          
    # add some text for labels, title and axes ticks
    ax.set_ylabel(yLabel)
    ax.set_title(fig_title)
    ax.set_xticks(ind+width)

    for rect in lst_rects:
        for rect_i in rect:
            height = rect_i.get_height()
            ax.text(rect_i.get_x()+rect_i.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')

    plt.show()
        


#----------------------
menMeans   = (20, 35, 30, 35, 27)
menStd     = (2, 3, 4, 1, 2)

womenMeans  = (25, 32, 34, 20, 25)
womenStd    = (3, 5, 2, 3, 3)

toddlersM   = [15,10,22,25,13]
toddlersS   = [3,2,4,5,1]

infantsM    = [15,15,15,15,15]
infantsS    = [4,5,2,3,1]

lst_bar_heights = [menMeans, womenMeans, toddlersM, infantsM]
lst_stdevs      = [menStd, womenStd, toddlersS, infantsS]
yLabel          = 'Scores'
xLabel          = ['G1', 'G2', 'G3', 'G4', 'G5']
fig_title       = 'Scores by group and gender'
bar_width       = 0.15
lst_legends     = ['Men', 'Women', 'Toddlers', 'Infants']
lst_bar_colors  = ['r','y','g','b']
 
#enforce that for every set of bars there's a corresponding set of stdevs, and a bar color
if ( (len(lst_bar_heights)!=len(lst_stdevs)) or (len(lst_bar_heights)!=len(lst_bar_colors)) or (len(lst_bar_heights)!=len(lst_bar_colors)) ):
    print "lst_bar_heights (len=%d), lst_stdevs (len=%d), lst_bar_colors (len=%d) must have the same number of elements" %(len(lst_bar_heights), len(lst_stdevs), len(lst_bar_colors))
    sys.exit()

if lst_legends:
    if not (len(lst_bar_heights)==len(lst_legends)):
        print "lst_bar_heights and lst_legends must have the same number of elements"
        sys.exit()

#enforce that for each bar height there's a corresponding stdev
for bar_height_set, stdev_set in zip(lst_bar_heights, lst_stdevs):
    if len(bar_height_set)!=len(stdev_set):
        print "lst_bar_height and lst_stdevs don't have the same length"
        print "bar height values (len=%d): %s" %(len(bar_height_set), bar_height_set)
        print "stdev values (len=%d): %s" %(len(stdev_set), stdev_set)
        sys.exit() 
 
fig, ax = plt.subplots()

barchart(ax, yLabel, xLabel, fig_title, bar_width, lst_bar_heights, lst_stdevs, lst_bar_colors, lst_legends=lst_legends)    

