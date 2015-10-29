#!/usr/bin/env python

from __future__ import print_function

import gc
import matplotlib
matplotlib.use('PDF')

from matplotlib.cbook import report_memory
import numpy as np
import matplotlib.pyplot as plt
# take a memory snapshot on indStart and compare it with indEnd

rand = np.random.rand

indStart, indEnd = 200, 401
mem_size, coll_count = [], []
for i in range(indEnd):

    fig = plt.figure(1)
    fig.clf()

    t1 = np.arange(0.0, 2.0, 0.01)
    y1 = np.sin(2 * np.pi * t1)
    y2 = rand(len(t1))
    X = rand(50, 50)

    ax = fig.add_subplot(221)
    ax.plot(t1, y1, '-')
    ax.plot(t1, y2, 's')

    ax = fig.add_subplot(222)
    ax.imshow(X)

    ax = fig.add_subplot(223)
    ax.scatter(rand(50), rand(50), s=100 * rand(50), c=rand(50))

    ax = fig.add_subplot(224)
    ax.pcolor(10 * rand(50, 50))

    fig.savefig('tmp%d' % i, dpi=75)
    plt.close(1)

    coll = gc.collect()
    val = report_memory(i)
    print(i, val)
    if i == indStart:
        start = val  # wait a few cycles for memory usage to stabilize
    mem_size.append(val)
    coll_count.append(coll)

end = val
print('Average memory consumed per loop: %1.4fk bytes\n' %
      ((end - start) / float(indEnd - indStart)))
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(mem_size, 'r')
ax.set_ylabel('memory size', color='r')
ax2.plot(coll_count, 'k')
ax2.set_ylabel('collect count', color='k')
fig.savefig('report')
