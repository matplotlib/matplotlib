"""
Matplotlib used to recompute autoscaled limits after every plotting (plot(),
bar(), etc.) call. It now only does so when actually rendering the canvas, or
when the user queries the Axes limits and that is possible through **autoscale**
feature.
This is an improvement over the previous case when user has to manually
autoscale the Axes(or axis) according to data.
This particular method is a part of matplotlib.axes.Axes class. It is used to
scale the Axes(or axis) according to data limits.   
Before autoscale one could have to manually struggle with the Axes to scale
itself according to data.   
Whenever we use autoscale method the axes(or axis if specified only one axis)
does recalculate its limits with respect to data.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch     
from matplotlib.path import Path 
from matplotlib.collections import EllipseCollection

# Utility function to describe enabling and disabling of autoscaling.
def autoscale():
    x = np.sin(np.linspace(0,6))
    fig,ax =plt.subplots(ncols=2)

    # Firstly we will explicitly enable autoscaling (However it is by-default enabled for pyplot artist.)
    ax[0].autoscale(True)
    ax[0].plot(x)

    # Now we disable it .
    ax[1].autoscale(False)
    ax[1].plot(x)

    fontdict ={
        "fontsize":15,
        "fontweight" : 14
    }

    ax[0].set_title("Autoscale Enabled",fontdict = fontdict)
    ax[1].set_title("Autoscale Disabled",fontdict = fontdict)
    fig.canvas.draw()


# Utility function to describe autoscaling with margins.
def autoscale_and_margins(autoscale=False):
    fig,ax = plt.subplots(ncols=2)
    
    t = np.arange(-np.pi/2,(3 * np.pi)/2,0.1)
    f_t = np.cos(t)
    
    for axis in [ax[0],ax[1]]:
        axis.plot(t,f_t,color="red")
        axis.axhline(y=0,color="magenta",alpha=0.7)
        axis.margins(0.2,0.2)
    
    fontdict ={
        "fontsize":15,
        "fontweight" : 14
    }

    ax[1].autoscale(tight=True)
    ax[0].set_title("Without Autoscale",fontdict=fontdict)
    ax[1].set_title("With Autoscale",fontdict=fontdict)
    plt.tight_layout()
    fig.canvas.draw()

# Utility function for describing relation between autoscale and Collections Class  
def autoscale_and_Collection():
    fig,ax = plt.subplots(ncols=2)
    x = np.arange(10) 
    y = np.arange(15) 
    X, Y = np.meshgrid(x, y) 
      
    XY = np.column_stack((X.ravel(), Y.ravel())) 

    # We have to add create different artist instances for different axis.
         
    ec_1 = EllipseCollection(10, 10, 5, units ='y', 
                          offsets = XY * 0.5,           # when autoscaling is enabled then axis is autoscaled to this offset. 
                          transOffset = ax[0].transData, 
                          cmap ="jet") 
    ec_1.set_array((X * Y).ravel()) 
    
    ec_2 = EllipseCollection(10, 10, 5, units ='y', 
                          offsets = XY * 0.5,           # when autoscaling is enabled then axis is autoscaled to this offset. 
                          transOffset = ax[1].transData, 
                          cmap ="jet")
    ec_2.set_array((X * Y).ravel()) 
    
    fontdict ={
        "fontsize":15,
        "fontweight" : 14
    }

    ax[0].add_collection(ec_1) 
    ax[1].add_collection(ec_2)
    ax[1].autoscale_view()
    
    ax[0].set_title("Without Autoscale",fontdict=fontdict)
    ax[1].set_title("With Autoscale",fontdict=fontdict)

    fig.canvas.draw()
     
# Utility function for describing relationship between autoscale and Patches.
def autoscale_and_patches():
    fig,ax = plt.subplots(ncols=2)
    
    vertices = [(0,0),(0,3),(1,0),(0,0)]
    codes = [Path.MOVETO] + [Path.LINETO]*2 + [Path.MOVETO]
    vertices = np.array(vertices,float)
    
    path_1 = Path(vertices,codes)
    patches_1 = PathPatch(path_1,facecolor="magenta",alpha=0.7)  
    
    # matplotlib.patches.Patch is Base class of PathPatch and it                                                           
    # does not support autoscaling.  
    
    # creating two because re-using of artists not supported.
    
    path_2 = Path(vertices,codes)
    patches_2 = PathPatch(path_2,facecolor="magenta",alpha=0.7)   
    
    fontdict ={
        "fontsize":15,
        "fontweight" : 14
    }

    ax[0].add_patch(patches_1)
    ax[1].add_patch(patches_2)
    ax[1].autoscale()

    ax[0].set_title("Without Autoscale",fontdict=fontdict)
    ax[1].set_title("With Autoscale",fontdict=fontdict)
    fig.canvas.draw()

# Utility function for describing case when we need to disable autoscaling.
def autoscale_disable():
    fig,ax = plt.subplots(ncols=2)

    x = np.arange(10)

    # Disable Autoscaling
    ax[1].autoscale(False) 
        
    for axis  in [ax[0],ax[1]]:
        axis.set_ylim(0,2)
        axis.plot(x)
        
    ax[0].set_title("Autoscale Enabled")
    ax[1].set_title("Autoscale Disabled")
    plt.tight_layout()
    
    fig,ax = plt.subplots()
    ax.plot(x)
    ax.set_title("Original Plot")
    plt.tight_layout()


"""
=============
autoscale
=============

There are some cases when we have to explicitly enable or disable autoscaling
feature and we would see some of those cases in the following tutorial. Lets
just discuss how we can explicitly enable or disable autoscaling.
"""

autoscale()

"""
=====================
autoscale and margins
=====================

Whenever we set margins our axes remains invariant of the change caused by
it.Hence we use autoscaling if we want data to be bound with the axes
irrespective of the margin set.
"""

autoscale_and_margins()

"""
=====================
Artist with autoscale
=====================

Collection and Patch subclasses of Artist class does not support autoscaling by
default. If one wants to enable autoscaling he have to explicitly enable it. See
Axes limits in the below plots.   

For further reference on relation of Autoscale feature with Collection Class
see-
https://matplotlib.org/3.2.1/api/prev_api_changes/api_changes_3.2.0/behavior.html#autoscaling
"""

"""
=========================
autoscale and Collection
=========================
Let's have a look at how autoscaling affects Collection instance. 
"""
autoscale_and_Collection()

"""
=====================
autoscale and patches
=====================
Let's have a look at how autoscaling affects PathPatch instance. 
"""
autoscale_and_patches()

"""
Some of the subclasses under Collection class are : 
https://matplotlib.org/3.1.1/_images/inheritance-1d05647d989bf64e3e438a24b19fee19432184da.png
reference site : https://matplotlib.org/3.1.1/api/collections_api.html
"""

"""
Till now we have seen how we can enable autoscaling. Now lets just discuss under
which case we possibly need to disable it.

Suppose we want to plot a line at 45 degrees to x and y axes and we want data to
be shown within a given range out of the whole and hence we have to set limits
for x and y axes and in that case we have to first disable autoscaling and then
we can set the limits.
"""

"""
=================
autoscale disable
=================
Let's have a look at above case.
"""

autoscale_disable()

"""
As we can see that setting the y_lim between 0 and 2 worked for both axes but
the one whose autoscaling was not disabled, got autoscaled in the x-axis and the
one which had autoscaling disabled, maintained the default range(between 0 to 1)
for x-axis.
"""