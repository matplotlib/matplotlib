import pygtk
pygtk.require('2.0')
import gtk
import time
from matplotlib.matlab import *

def get_memory():
    "Simulate a function that returns system memory"
    return 100*(0.5+0.5*sin(0.5*pi*time.time()))

def get_cpu():
    "Simulate a function that returns cpu usage"
    return 100*(0.5+0.5*sin(0.2*pi*(time.time()-0.25)))

def get_net():
    "Simulate a function that returns network bandwidth"
    return 100*(0.5+0.5*sin(0.7*pi*(time.time()-0.1)))

def get_stats():
    return get_memory(), get_cpu(), get_net()

fig = figure(1)
ax = subplot(111)
ind = arange(1,4)
pm, pc, pn = bar(ind, get_stats())
centers = ind + 0.5*pm.get_width()
pm.set_facecolor('r')
pc.set_facecolor('g')
pn.set_facecolor('b')
ax.set_xlim([0.5,4])
ax.set_xticks(centers)
ax.set_ylim([0,100])
ax.set_xticklabels(['Memory', 'CPU', 'Bandwidth'])
ax.set_ylabel('Percent usage')
ax.set_title('System Monitor')

manager = get_current_fig_manager()
def updatefig(*args):
    m,c,n = get_stats()

    pm.set_height(m)
    pc.set_height(c)
    pn.set_height(n)
    ax.set_ylim([0,100])

    manager.canvas.draw()

    return gtk.TRUE


gtk.timeout_add(250, updatefig)
show()
    


