#!/usr/bin/env python

import os, sys, time
import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('Agg')
from matplotlib.matlab import *


def report_memory(i):
    pid = os.getpid()
    a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
    print i, '  ', a2[1],
    return int(a2[1].split()[1])



# take a memory snapshot on indStart and compare it with indEnd
indStart, indEnd = 30, 150
for i in range(indEnd):
    ind = arange(100)
    xx = rand(len(ind))

    figure(1)
    subplot(221)
    plot(ind, xx)

    subplot(222)
    X = rand(50,50)
    
    imshow(X)
    subplot(223)
    scatter(rand(50), rand(50))
    subplot(224)
    pcolor(10*rand(50,50))

    savefig('tmp%d' % i, dpi = 75)
    #fd = file('tmp%d' % i, 'wb')
    #savefig(fd, dpi = 75)
    #fd.close()
    close(1)


    val = report_memory(i)
    if i==indStart: start = val # wait a few cycles for memory usage to stabilize

end = val
print 'Average memory consumed per loop: %1.4fk bytes\n' % ((end-start)/float(indEnd-indStart))

"""
Average memory consumed per loop: 0.0053k bytes
"""
