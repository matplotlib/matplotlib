#!/usr/bin/env python

import matplotlib.numerix as nx
from matplotlib.mlab import linspace
import matplotlib.cm as cm
from pylab import figure, show, rc


# force square figure and square axes looks better for polar, IMO
fig = figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

N = 20
theta = nx.arange(0.0, 2*nx.pi, 2*nx.pi/N)
radii = 10*nx.mlab.rand(N)
width = nx.pi/4*nx.mlab.rand(N)
bars = ax.bar(theta, radii, width=width, bottom=0.1)
for r,bar in zip(radii, bars):
    bar.set_facecolor( cm.jet(r/10.))
    bar.set_alpha(0.5)

show()
