#!/usr/bin/env python
"""
pcolormesh uses a QuadMesh, a faster generalization of pcolor, but
with some restrictions.

This demo illustrates a bug in quadmesh with masked data.
"""

import numpy as npy
from matplotlib.pyplot import figure, show, savefig
from matplotlib import cm, colors
from numpy import ma

n = 56
x = npy.linspace(-1.5,1.5,n)
y = npy.linspace(-1.5,1.5,n*2)
X,Y = npy.meshgrid(x,y);
Qx = npy.cos(Y) - npy.cos(X)
Qz = npy.sin(Y) + npy.sin(X)
Qx = (Qx + 1.1)
Z = npy.sqrt(X**2 + Y**2)/5;
Z = (Z - Z.min()) / (Z.max() - Z.min())

# The color array can include masked values:
Zm = ma.masked_where(npy.fabs(Qz) < 0.5*npy.amax(Qz), Z)


fig = figure()
ax = fig.add_subplot(121)
ax.set_axis_bgcolor("#bdb76b")
ax.pcolormesh(Qx,Qz,Z)
ax.set_title('Without masked values')

ax = fig.add_subplot(122)
ax.set_axis_bgcolor("#bdb76b")
#  You can control the color of the masked region:
#cmap = cm.jet
#cmap.set_bad('r', 1.0)
#ax.pcolormesh(Qx,Qz,Zm, cmap=cmap)
#  Or use the default, which is transparent:
col = ax.pcolormesh(Qx,Qz,Zm)
ax.set_title('With masked values')
show()

savefig("quadmesh_demo")
