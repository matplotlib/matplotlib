#!/usr/bin/env python
"""
pcolormesh uses a QuadMesh, a faster generalization of pcolor, but
with some restrictions.
"""

import numpy as nx
from pylab import figure,show
from matplotlib import cm, colors

n = 56
x = nx.linspace(-1.5,1.5,n)
X,Y = nx.meshgrid(x,x);
Qx = nx.cos(Y) - nx.cos(X)
Qz = nx.sin(Y) + nx.sin(X)
Qx = (Qx + 1.1)
Z = nx.sqrt(X**2 + Y**2)/5;
Z = (Z - nx.amin(Z)) / (nx.amax(Z) - nx.amin(Z))

# The color array can include masked values:
Zm = nx.ma.masked_where(nx.fabs(Qz) < 0.5*nx.amax(Qz), Z)


fig = figure()
ax = fig.add_subplot(121)
ax.pcolormesh(Qx,Qz,Z)
ax.set_title('Without masked values')

ax = fig.add_subplot(122)
#  You can control the color of the masked region:
#cmap = cm.jet
#cmap.set_bad('r', 1.0)
#ax.pcolormesh(Qx,Qz,Zm, cmap=cmap)
#  Or use the default, which is transparent:
col = ax.pcolormesh(Qx,Qz,Zm)
ax.set_title('With masked values')
show()

