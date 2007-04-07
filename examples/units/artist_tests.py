"""
Test unit support with each of the matplotlib primitive artist types

The axes handles unit conversions and the artists keep a pointer to
their axes parent, so you must init the artists with the axes instance
if you want to initialize them with unit data, or else they will not
know how to convert the units to scalars
"""
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.text as text
import matplotlib.collections as collections
import matplotlib.units as units

from basic_units import cm, inch

from pylab import figure, show, nx

fig = figure()
ax = fig.add_subplot(111)
ax.xaxis.set_units(cm)
ax.yaxis.set_units(cm)

line = lines.Line2D([0*cm, 1.5*cm], [0*cm, 2.5*cm], lw=2, color='black', axes=ax)
ax.add_line(line)

rect = patches.Rectangle( (1*cm, 1*cm), width=5*cm, height=2*cm, alpha=0.2, axes=ax)
ax.add_patch(rect)

ax.set_xlim(-1*cm, 10*cm)
ax.set_ylim(-1*cm, 10*cm)
ax.xaxis.set_units(inch)
show()
