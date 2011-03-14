#!/usr/bin/env python

"""
Another broken test...
"""

from __future__ import print_function
import os, sys, time
import matplotlib.nxutils as nxutils
from numpy.random import rand

def report_memory(i):
    pid = os.getpid()
    a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
    print(i, '  ', a2[1], end='')
    return int(a2[1].split()[1])



for i in range(500):
    report_memory(i)
    verts = rand(100, 2)
    b = nxutils.pnpoly(x, y, verts)   # x, y don't exist

    for i in range(500):
        report_memory(i)
        verts = rand(100, 2)
        points = rand(10000,2)
        mask = nxutils.points_inside_poly(points, verts)

