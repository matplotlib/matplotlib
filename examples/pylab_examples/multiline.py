#!/usr/bin/env python
from pylab import *
#from matplotlib.pyplot import *
#from numpy import arange

if 1:
    figure(figsize=(7, 4))
    ax = subplot(121)
    ax.set_aspect(1)
    plot(arange(10))
    xlabel('this is a xlabel\n(with newlines!)')
    ylabel('this is vertical\ntest', multialignment='center')
    #ylabel('this is another!')
    text(2, 7,'this is\nyet another test',
         rotation=45,
         horizontalalignment = 'center',
         verticalalignment   = 'top',
         multialignment      = 'center')

    grid(True)



    subplot(122)
    
    text(0.29, 0.7, "Mat\nTTp\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

    text(0.34, 0.7, "Mag\nTTT\n123", size=18,
         va="baseline", ha="left", multialignment="left",
         bbox=dict(fc="none"))

    text(0.95, 0.7, "Mag\nTTT$^{A^A}$\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

    xticks([0.2, 0.4, 0.6, 0.8, 1.],
           ["Jan\n2009","Feb\n2009","Mar\n2009", "Apr\n2009", "May\n2009"])
    
    axhline(0.7)
    title("test line spacing for multiline text")

subplots_adjust(bottom=0.25, top=0.8)
draw()
show()
