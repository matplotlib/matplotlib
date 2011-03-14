"""
Broken.
"""

from __future__ import print_function
import os, sys, time

import matplotlib.nxutils as nxutils
from numpy.random import rand
import matplotlib.mlab
import matplotlib.patches as patches
if 1:
    numtrials, numverts, numpoints = 50, 1000, 1000
    verts = patches.CirclePolygon((0.5, 0.5), radius=0.5, resolution=numverts).get_verts()

    t0 = time.time()
    for i in range(numtrials):
        points = rand(numpoints,2)
        mask = matplotlib.mlab._inside_poly_deprecated(points, verts)
                    ### no such thing
    told = time.time() - t0

    t0 = time.time()
    for i in range(numtrials):
        points = rand(numpoints,2)
        mask = nxutils.points_inside_poly(points, verts)
    tnew = time.time() - t0
    print(numverts, numpoints, told, tnew, told/tnew)



