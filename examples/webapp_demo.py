#!/usr/bin/env python
# This example shows how to use the agg backend directly to create
# images, which may be of use to web application developers who want
# full control over their code without using the matlab interface to
# manage figures, figure closing etc.
#
# The rc command is used to create per-script default figure
# customizations of the rc parameters; see
# http://matplotlib.sf.net/.matplotlibrc .  You may prefer to set the
# rc parameters in the rc file itself.  Note that you can keep
# directory level default configurations by placing different rc files
# in the directory that the script runs in.
#
# I am making no effort here to make a figure that looks good --
# rather I am just trying to show the various ways to use matplotlib
# to customize your figure using the matplotlib API

import matplotlib
matplotlib.use('Agg')  # force the antigrain backend
from matplotlib import rc
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.cbook import iterable
import matplotlib.numerix as nx

# set some global properties that affect the defaults of all figures
rc('lines', linewidth=2)              # thicker plot lines
rc('grid', color=0.75, linestyle='-') # solid gray grid lines
rc('axes', hold=True,                 # hold state is on
   grid=True, facecolor='y')          # yellow background, grid on
rc('tick', color='r', labelsize=20)   # big red ticks

def setapi(o, **kwargs):
    """
    for all key, value pairs in kwargs, and all objects in (possibly)
    iterable o, look for a method o.set_key and try to call
    o.set_key(value) if it exists.  This is basically a refinition of
    the matlab interface set command
    """
    if not iterable(o): o = [o]
    for thiso in o: # iterate over the objects
        for k,v in kwargs.items():
            func = getattr(thiso, 'set_'+k)
            if func is None: continue
            func(v)

def make_fig():
    """
    make a figure

    No need to close figures or clean up since the objects will be
    destroyed when they go out of scope
    """
    fig = Figure()
    #ax = fig.add_subplot(111)  # add a standard subplot

    # add an axes at left, bottom, width, height; by making the bottom
    # at 0.3, we save some extra room for tick labels
    ax = fig.add_axes([0.2, 0.3, 0.7, 0.6])

    line,  = ax.plot([1,2,3], 'ro--', markersize=12, markerfacecolor='g')

    # make a translucent scatter collection 
    x = nx.rand(100)
    y = nx.rand(100)
    area = nx.pi*(10 * nx.rand(100))**2 # 0 to 10 point radiuses
    c = ax.scatter(x,y,area)
    c.set_alpha(0.5)  

    # add some text decoration
    ax.set_title('My first image')
    ax.set_ylabel('Some numbers')    
    ax.set_xticks( (.2,.4,.6,.8) )
    labels = ax.set_xticklabels(('Bill', 'Fred', 'Ted', 'Ed'))

    # To set object properties, you can either iterate over the
    # objects manually, or define you own set command, as in setapi
    # above.
    #setapi(labels, rotation=45, fontsize=12)
    for l in labels:
        l.set_rotation(45)
        l.set_fontsize(12)
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure('webapp.png', dpi=150)

make_fig()
