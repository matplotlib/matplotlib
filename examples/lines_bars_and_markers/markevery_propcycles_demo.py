"""
===============================================================
Implementation of prop_cycle support for the markevery property
===============================================================

An example demonstrating the fix to issue #8576, enabling full utilization of
the markevery property in axes.prop_cycle assignments. Uses the same list
of cases from the examples/pylab_examples/markevery_demo.py, implemented
as an rcParams assignment as a composite cycler after addition with a colormap.

Renders an ascending series of Weierstrass functions from a 2D Numpy array
"""


from __future__ import division
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# define a list of markevery cases to plot
cases = [None,
         8,
         (30, 8),
         [16, 24, 30], [0, -1],
         slice(100, 200, 3),
         0.1, 0.3, 1.5,
         (0.0, 0.1), (0.45, 0.1)]
        
markevery_cycler = cycler(markevery=cases) 

#compose cyclic color scheme along with markevery cycler
phases = np.linspace(0, 1, len(cases))
colorlist = [ plt.cm.jet(phase) for phase in phases]
color_cycler = cycler(color=colorlist)

composite_cycler = markevery_cycler + color_cycler
mpl.rcParams['axes.prop_cycle']=composite_cycler


# define the data for cartesian plots
numExamples =2*len(cases)+1

numSamples =201

x = np.linspace(-np.pi/2, np.pi/2, numSamples)
yarray = np.zeros((numExamples, numSamples))
yarray[0] = np.sum(np.power(0.5,n)*np.cos(np.pi*x*np.power(8,n)) for \
n in np.arange(0,100,1))

for i in range(1, numExamples):
    yarray[i]=yarray[i-1]+1

# plot each markevery case for linear x and y scales
fig = plt.figure()
ax = fig.add_subplot(111) 
for i, y in enumerate(yarray):
    ax.plot(x, y, 'o', markersize=1.5, ls='-', linewidth=0.4)
lgd = plt.legend(title='markevery values', handles=[mpl.patches.Patch(color=colorlist[i], label=str(cases[i])) \
for i in range(len(cases)) ], bbox_to_anchor=(1.0,1.02),loc=2)

plt.title("Weierstrass functions with 'markevery' prop_cycles enabled")
#plt.savefig('markevery_composite', bbox_extra_artists=(lgd,), bbox_inches='tight', format="png")
#plt.savefig('markevery_composite', bbox_extra_artists=(lgd,), bbox_inches='tight', format="pdf")
#plt.savefig('markevery_composite', bbox_extra_artists=(lgd,), bbox_inches='tight', format="svg")
plt.show()