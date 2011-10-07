#!/usr/bin/env python
"""Demonstrate the ternary projections for matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import rcParams
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# Test with the default rcParams.
rcParams['figure.subplot.left'] = 0.125 # the left side of the subplots of the figure
rcParams['figure.subplot.right'] = 0.9 # the right side of the subplots of the figure
rcParams['figure.subplot.bottom'] = 0.1 # the bottom of the subplots of the figure
rcParams['figure.subplot.top'] = 0.9 # the top of the subplots of the figure


# Example 1 -- Basic points and lines
fig = plt.figure()
axab = fig.add_subplot(111, projection='ternaryab')
axab.set_title("Points and Lines on a Ternary Plot")
#axab.grid(False) # Turn the grid off. (It's on by default.)
#axab.grid() # Toggle the grid back on.
axab.plot(1/3.0, 1/3.0, 'ko', label="Equal parts")
axab.plot([0.0, 1.0], [0.5, 0.0], 'r:', label="Equal B & C")
axab.plot([0.0, 0.5], [1.0, 0.0], 'g:', label="Equal A & C")
axab.plot([0.0, 0.5], [0.0, 0.5], 'b:', label="Equal A & B")
axab.plot([0.0, 0.0], [0.0, 1.0], 'r--', label="No A", lw=3.0)
axab.plot([0.0, 1.0], [0.0, 0.0], 'g--', label="No B", lw=3.0)
axab.plot([0.0, 1.0], [1.0, 0.0], 'b--', label="No C", lw=3.0)
axab.annotate("All\nA", (1, 0), ha='center', va='center', xytext=(0.9,0.05),
            textcoords='data', arrowprops=dict(width=0.2, headwidth=8.0,
            shrink=0.05))
axab.annotate("All\nB", (0, 1), ha='center', va='center', xytext=(0.1,0.8),
            textcoords='data', arrowprops=dict(width=0.2, headwidth=8.0,
            shrink=0.05))
axab.annotate("All\nC", (0, 0), ha='center', va='center', xytext=(0.1,0.1),
            textcoords='data', arrowprops=dict(width=0.2, headwidth=8.0,
            shrink=0.05))
axab.set_xlabel("Fraction of A")
axab.set_ylabel("Fraction of B")
axab.set_zlabel("Fraction of C")
axab.legend()
axab.set_alabel("Component A")
axab.set_blabel("Component B")
axab.set_clabel("Component C")

# Example 3 -- Alternative indexing
fig = plt.figure()
axab = fig.add_subplot(111, projection='ternaryab')
axab.plot(0.9, 0.05, 'ro', label="Mostly A (via AB)")
axab.plot(0.05, 0.9, 'go', label="Mostly B (via AB)")
axab.plot(0.05, 0.05, 'bo', label="Mostly C (via AB)")

axbc = axab.twinx()
axbc.plot(0.1, 0.1, 'ro', label="Mostly A (via BC)")
axbc.plot(0.8, 0.1, 'go', label="Mostly B (via BC)")
axbc.plot(0.1, 0.8, 'bo', label="Mostly C (via BC)")

axca = axab.twiny()
axca.plot(0.15, 0.7, 'ro', label="Mostly A (via CA)")
axca.plot(0.15, 0.15, 'go', label="Mostly B (via CA)")
axca.plot(0.7, 0.15, 'bo', label="Mostly C (via CA)")
axca.set_title("Test of Indexing")
axca.set_ylabel("ca's y")

# Example 4 -- Patches, fills, and outlines
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

# Create the plot.
fig = plt.figure()
axab = fig.add_subplot(111, projection='ternaryab')
axab.add_patch(PathPatch(path, facecolor='r', alpha=0.5))
axab.plot([0.8, 0.6, 0.6, 0.8], [0.1, 0.3, 0.1, 0.1], 'k')
axab.fill_between(a, b, y2=0.33, color='b')
axab.plot(a, b, 'w')
axab.set_title('Patches, Fills, and Outlines')


# Example 2 -- Scatter plot of soil data
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
fig = plt.figure()
axab = fig.add_subplot(111, projection='ternaryab')
s = axab.scatter(x=soil['sand'], y=soil['silt'], c=soil['organic matter'],
                 s=100.0*soil['porosity'], marker='o')
axab.set_title("Soil Plot\n(The sizes of markers indicate porosity.)")
#axab.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0]) # **Why won't this work?
axab.set_xlabel("Sand / 1")
axab.set_ylabel("Silt / 1")
#axab.set_rlabel("Clay / 1")
cax = axab.colorbar(mappable=s)
cax.set_label("Organic Matter / 1")


# Finish.
plt.show()
