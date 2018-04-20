#!/usr/bin/env python
"""Demonstrate the ternary projections for matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.ternary import Ternary
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# Example 1 -- Basic points and lines
ter = Ternary()
#ter.ab.set_xlim(0, 10)
#ter.ab.invert_xaxis() **Can this be used to change from ccw-increasing axes to
# cw-increasing axes?
ter.set_title("Points and Lines on a Ternary Plot")
#ter.set_axis_bgcolor('b')
ter.ab.plot(1/3.0, 1/3.0, 'ko', label="Equal parts")
ter.ab.plot([0.0, 1.0], [0.5, 0.0], 'r:', label="Equal B & C")
ter.ab.plot([0.0, 0.5], [1.0, 0.0], 'g:', label="Equal A & C")
ter.ab.plot([0.0, 0.5], [0.0, 0.5], 'b:', label="Equal A & B")
ter.ab.plot([0.0, 0.0], [0.0, 1.0], 'r--', label="No A", lw=3.0)
ter.bc.plot([0.0, 0.0], [0.0, 1.0], 'g--', label="No B", lw=3.0)
ter.ca.plot([0.0, 0.0], [0.0, 1.0], 'b--', label="No C", lw=3.0)
ter.ab.annotate("All\nA", (1, 0), ha='center', va='center', xytext=(0.8,0.1),
                 textcoords='data', arrowprops=dict(facecolor='k', width=0.2,
                 headwidth=8.0, shrink=0.05))
ter.bc.annotate("All\nB", (1, 0), ha='center', va='center', xytext=(0.8,0.1),
                 textcoords='data', arrowprops=dict(facecolor='k', width=0.2,
                 headwidth=8.0, shrink=0.05))
ter.ca.annotate("All\nC", (1, 0), ha='center', va='center', xytext=(0.8,0.1),
                 textcoords='data', arrowprops=dict(facecolor='k', width=0.2,
                 headwidth=8.0, shrink=0.05))
ter.ab.set_xlabel("Relative part of A")
ter.bc.set_xlabel("Relative part of B")
ter.ca.set_xlabel("Relative part of C")
ter.ab.set_tiplabel("Comp. A")
ter.bc.set_tiplabel("Comp. B")
ter.ca.set_tiplabel("Comp. C")
ter.legend()
#print ter.ab.viewLim.x0
#print ter.ab.viewLim.intervalx
#print ter.ab.viewLim.intervaly
#print ter.ab.get_xlim()
#print ter.ab.get_autoscale_on()
#print ter.ab.get_autoscalex_on()


# Example 2 -- Alternative indexing
ter = Ternary()
ter.set_title("Alternative Indexing\n"
              "(Like-colored points should radiate from the center.)")
ter.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ter.ab.plot(0.9, 0.05, 'ro', label="Mostly A (via a, b)")
ter.ab.plot(0.05, 0.9, 'go', label="Mostly B (via a, b)")
ter.ab.plot(0.05, 0.05, 'bo', label="Mostly C (via a, b)")
ter.ab.set_xlabel("x-label of axes ab")
ter.ab.set_tiplabel("Comp. A", tipoffset=0.18)

ter.bc.plot(0.1, 0.1, 'r^', label="Mostly A (via b, c)")
ter.bc.plot(0.8, 0.1, 'g^', label="Mostly B (via b, c)")
ter.bc.plot(0.1, 0.8, 'b^', label="Mostly C (via b, c)")
ter.bc.set_xlabel("x-label of axes bc")
ter.bc.set_tiplabel("Comp. B", tipoffset=0.18)

ter.ca.plot(0.15, 0.7, 'rs', label="Mostly A (via c, a)")
ter.ca.plot(0.15, 0.15, 'gs', label="Mostly B (via c, a)")
ter.ca.plot(0.7, 0.15, 'bs', label="Mostly C (via c, a)")
ter.ca.set_xlabel("x-label of axes ca")
ter.ca.set_tiplabel("Comp. C", tipoffset=0.18)

ter.legend()


# Example 3 -- Manual vs. automatic setup

# Define the path for the patch.
vertices = [
    (0.1, 0.1),
    (0.1, 0.8),
    (0.8, 0.1),
    (0.1, 0.1),
    (0., 0.)] # ignored
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY]
path = Path(vertices, codes)

# Define some arbitrary data.
n_cycles = 10
t = np.pi*np.array(range(n_cycles*100))/50.0
a = 1.0/3.0 + np.exp(-t/10.0)*np.sin(t)/6.0
b = 1.0/3.0 + np.exp(-t/10.0)*np.cos(t)/6.0

# Set up the figure.
fig = plt.figure()
fig.subplots_adjust(wspace=0.45)
fig.suptitle("Manual vs. Automatic Setup\n(should give the same results)", y=0.92)

# Example 3a -- Manual approach
axab = fig.add_subplot(121, projection='ternaryab')
axbc = axab.twinx(projection='ternarybc')
axca = axab.twinx(projection='ternaryca')
axab.set_title('Manual', y=1.05)

axab.add_patch(PathPatch(path, facecolor='g', alpha=0.5))
axab.fill_between(a, b, y2=0.33, color='b')
axab.plot(a, b, 'w')
axab.set_tiplabel("Comp. 1", tipoffset=0.18)
axbc.set_tiplabel("Comp. 2", tipoffset=0.18)
axca.set_tiplabel("Comp. 3", tipoffset=0.18)
axab.grid(False) # Turn the grid off. (It's on by default.)
axbc.grid(False)
axca.grid(False)

# Example 3b -- Automatic approach
ter = Ternary(fig.add_subplot(122, projection='ternaryab'))
ter.set_title("Automatic", y=1.05)
ter.grid(False) # Turn the grid off. (It's on by default.)

ter.ab.add_patch(PathPatch(path, facecolor='g', alpha=0.5))
ter.ab.fill_between(a, b, y2=0.33, color='b')
ter.ab.plot(a, b, 'w')
ter.ab.set_tiplabel("Comp. 1", tipoffset=0.18)
ter.bc.set_tiplabel("Comp. 2", tipoffset=0.18)
ter.ca.set_tiplabel("Comp. 3", tipoffset=0.18)


import matplotlib.colorbar as mcolor
# Example 4 -- Scatter plot of soil data
# Define the data (based on that by C. P. H. Lewis as triangleplot_demo.csv,
# http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg10051.html).
soil = {'sand': np.array([0.82, 0.17, 0.8 , 0.63, 0.5 , 0.0 , 0.3 , 0.3 , 0.73,
                          0.03, 0.77]),
        'silt': np.array([0.04, 0.11, 0.14, 0.3 , 0.4 , 0.81, 0.41, 0.33, 0.03,
                          0.37, 0.21]),
        'clay': np.array([0.14, 0.72, 0.06, 0.07, 0.1 , 0.19, 0.29, 0.37, 0.24,
                          0.60, 0.02]),
        'organic matter': np.array([0.45, 0.59, 0.35, 0.41, 0.07, 0.22, 0.31,
                                    0.21, 0.4 , 0.27, 0.2 ]),
        'porosity': np.array([0.48, 0.32, 0.23, 0.22, 0.62, 0.35, 0.43, 0.27,
                              0.44, 0.23, 0.53]),
        'site': ('Ger', 'Ma', 'Ss', 'Ss2', 'Din', 'Pro', 'Esn', 'Tra', 'Ffg',
                  'Rob', 'Aqw')}

# Create the plot.
ter = Ternary()
ter.ab._set_total(100)
ter.bc._set_total(100)
ter.ca._set_total(100)
s = ter.ab.scatter(x=100*soil['sand'], y=100*soil['silt'], c=100*soil['organic matter'],
                   cmap=cm.BuGn, s=100.0*soil['porosity'], marker='o')
ter.set_title("Soil Plot\n(The sizes of the markers indicate porosity.)")
ter.ab.set_tiplabel("Sand", tipoffset=0.12)
ter.bc.set_tiplabel("Silt", tipoffset=0.12)
ter.ca.set_tiplabel("Clay", tipoffset=0.12)
ter.ab.set_xlabel("Relative part / %")
ter.bc.set_xlabel("Relative part / %")
ter.ca.set_xlabel("Relative part / %")
cax = ter.colorbar(mappable=s)
cax.set_label("Organic Matter / %")

# Finish.
plt.show()
