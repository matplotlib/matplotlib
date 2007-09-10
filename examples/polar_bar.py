#!/usr/bin/env python

import numpy as npy
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show, rc


# force square figure and square axes looks better for polar, IMO
fig = figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

N = 20
theta = npy.arange(0.0, 2*npy.pi, 2*npy.pi/N)
radii = 10*npy.random.rand(N)
width = npy.pi/4*npy.random.rand(N)
bars = ax.bar(theta, radii, width=width, bottom=0.1)
for r,bar in zip(radii, bars):
    bar.set_facecolor( cm.jet(r/10.))
    bar.set_alpha(0.5)

show()
