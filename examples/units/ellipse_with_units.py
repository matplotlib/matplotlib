"""
Compare the ellipse generated with arcs versus a polygonal approximation
"""
from basic_units import cm
import numpy as npy
from matplotlib import patches
from pylab import figure, show

xcenter, ycenter = 0.38*cm, 0.52*cm
#xcenter, ycenter = 0., 0.
width, height = 1e-1*cm, 3e-1*cm
angle = -30

theta = npy.arange(0.0, 360.0, 1.0)*npy.pi/180.0
x = 0.5 * width * npy.cos(theta)
y = 0.5 * height * npy.sin(theta)

rtheta = angle*npy.pi/180.
R = npy.array([
    [npy.cos(rtheta),  -npy.sin(rtheta)],
    [npy.sin(rtheta), npy.cos(rtheta)],
    ])


x, y = npy.dot(R, npy.array([x, y]))
x += xcenter
y += ycenter

fig = figure()
ax = fig.add_subplot(211, aspect='auto')
ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow', linewidth=1, zorder=1)

e1 = patches.Ellipse((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)

ax.add_patch(e1)

ax = fig.add_subplot(212, aspect='equal')
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
e2 = patches.Ellipse((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)


ax.add_patch(e2)

#fig.savefig('ellipse_compare.png')
fig.savefig('ellipse_compare')

fig = figure()
ax = fig.add_subplot(211, aspect='auto')
ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow', linewidth=1, zorder=1)

e1 = patches.Arc((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)

ax.add_patch(e1)

ax = fig.add_subplot(212, aspect='equal')
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
e2 = patches.Arc((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)


ax.add_patch(e2)

#fig.savefig('arc_compare.png')
fig.savefig('arc_compare')

show()
