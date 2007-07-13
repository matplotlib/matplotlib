#!/usr/bin/env python

import os, sys, time, gc
import matplotlib
matplotlib.use('Agg')

from matplotlib.cbook import report_memory
import matplotlib.numerix as nx
from pylab import figure, show, close

# take a memory snapshot on indStart and compare it with indEnd

rand = nx.mlab.rand

indStart, indEnd = 200, 401
for i in range(indEnd):

    fig = figure(1)
    fig.clf()


    t1 = nx.arange(0.0, 2.0, 0.01)
    y1 = nx.sin(2*nx.pi*t1)
    y2 = rand(len(t1))
    X = rand(50,50)

    ax = fig.add_subplot(221)
    ax.plot(t1, y1, '-')
    ax.plot(t1, y2, 's')


    ax = fig.add_subplot(222)
    ax.imshow(X)

    ax = fig.add_subplot(223)
    ax.scatter(rand(50), rand(50), s=100*rand(50), c=rand(50))

    ax = fig.add_subplot(224)
    ax.pcolor(10*rand(50,50))

    fig.savefig('tmp%d' % i, dpi = 75)
    close(1)

    gc.collect()
    val = report_memory(i)
    print i, val
    if i==indStart: start = val # wait a few cycles for memory usage to stabilize

end = val
print 'Average memory consumed per loop: %1.4fk bytes\n' % ((end-start)/float(indEnd-indStart))

