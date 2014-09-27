#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt



def two_scales(ax1, ax2, time, data1, data2, param1_dic , param2_dic):
    """


    Demonstrate how to do two plots on the same axes with different left
    right scales.
    
    
    The trick is to use *2 different axes*.  Turn the axes rectangular
    frame off on the 2nd axes to keep it from obscuring the first.
    Manually set the tick locs and labels as desired.  You can use
    separate matplotlib.ticker formatters and locators as desired since
    the two axes are independent.
    
    This is achieved in the following example by calling the Axes.twinx()
    method, which performs this work. See the source of twinx() in
    axes.py for an example of how to do it for different x scales. (Hint:
    use the xaxis instance and call tick_bottom and tick_top in place of
    tick_left and tick_right.)
    
    The twinx and twiny methods are also exposed as pyplot functions.

    Parameters
    ----------
    ax : (type of axis)
        A description of axis

    data1: (first dataset)
        A description of data1
        
    data2 : (first dataset)
        A description of data2
        
    param_dic : This is a dictionary of the parameters of the style and color e.g. {line style: '-', text color = 'r'}
    Returns
    -------
    Overlays
    data1 : (Plot first data set)
    data2 : (Plot second data set)

     """
    def color_y_axes(ax, color):
        """Color your axes."""
        for t in ax.get_yticklabels():
            t.set_color(color)
        return None  
    

    ax1.plot(time, data1, param1_dic['color'] + param1_dic['style'])
    ax1.set_xlabel('time (s)')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('exp', color=param1_dic['color'])
    color_y_axes(ax1, param1_dic['color'])
   

    ax2.plot(time, data2, param2_dic['color'] + param2_dic['style'])
    ax2.set_ylabel('sin', color=param2_dic['color'])
    color_y_axes(ax2, param2_dic['color'])
    return plt.show()

#Create some mock data
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
s2 = np.sin(2*np.pi*t)

#Specify your parameter dictionary
d1 = {'style': '-', 'color' : 'r'}
d2 = {'style': '.', 'color' :'b'}

#create your axes
fig, ax = plt.subplots()
ax2 = ax.twinx()

#Call the function
two_scales(ax, ax2, t, s1, s2, d1, d2)




