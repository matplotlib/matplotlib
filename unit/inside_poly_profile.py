import os, sys, time

import matplotlib.nxutils as nxutils
import matplotlib.numerix as nx
import matplotlib.mlab
import matplotlib.patches as patches
if 1:
    numtrials, numverts, numpoints = 50, 1000, 1000
    verts = patches.CirclePolygon((0.5, 0.5), radius=0.5, resolution=numverts).get_verts()

    t0 = time.time()
    for i in range(numtrials):
        points = nx.mlab.rand(numpoints,2)
        mask = matplotlib.mlab._inside_poly_deprecated(points, verts)
    told = time.time() - t0

    t0 = time.time()
    for i in range(numtrials):
        points = nx.mlab.rand(numpoints,2)
        mask = nxutils.points_inside_poly(points, verts)
    tnew = time.time() - t0
    print numverts, numpoints, told, tnew, told/tnew



