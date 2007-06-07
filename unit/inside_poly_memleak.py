#!/usr/bin/env python

import os, sys, time
import matplotlib.nxutils as nxutils
import matplotlib.numerix as nx

def report_memory(i):
    pid = os.getpid()
    a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
    print i, '  ', a2[1],
    return int(a2[1].split()[1])



for i in range(500):
    report_memory(i)
    verts = nx.mlab.rand(100, 2)
    b = nxutils.pnpoly(x, y, verts)

    for i in range(500):
        report_memory(i)
        verts = nx.mlab.rand(100, 2)
        points = nx.mlab.rand(10000,2)
        mask = nxutils.points_inside_poly(points, verts)

