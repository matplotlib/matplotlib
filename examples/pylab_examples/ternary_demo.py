#!/usr/bin/env python
"""Demonstrate the Ternary class.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import rcParams
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# **Clean up these examples and reimplement one or more examples from C. P. H.
# Lewis, 2008-2009, http://nature.berkeley.edu/~chlewis/Sourcecode.html.

# Test with the default rcParams.
rcParams['figure.subplot.left'] = 0.125  # the left side of the subplots of the figure
rcParams['figure.subplot.right'] = 0.9    # the right side of the subplots of the figure
rcParams['figure.subplot.bottom'] = 0.1    # the bottom of the subplots of the figure
rcParams['figure.subplot.top'] = 0.9    # the top of the subplots of the figure


# Data 1 (arbitrary)
n_cycles = 10
t = np.pi*np.array(range(n_cycles*100))/50.0
b = 1.0/3.0 + np.exp(-t/10.0)*np.sin(t)/6.0
l = 1.0/3.0 + np.exp(-t/10.0)*np.cos(t)/6.0


# Example 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='ternary')
ax.set_title("Example of a Ternary Plot")
#ax.grid(True) # Turn the grid off. (It's on by default.)
#ax.grid(False) # Turn the grid off. (It's on by default.)
#ax.grid() # Toggle the grid back on.
ax.plot(b=1.0/3.0, l=1.0/3.0, fmt="kx", markersize=10, label="center")
ax.plot(b=[0, 0], l=[0, 1], fmt="r--", label="Edge (b = 0)", lw=4.0)
ax.plot(l=[0, 0], r=[0, 1], fmt="g--", label="Edge (l = 0)", lw=4.0)
ax.plot(r=[0, 0], b=[0, 1], fmt="b--", label="Edge (r = 0)", lw=4.0)
ax.plot(1, 0, 0, "rH", markersize=15, label="Vertex (1, 0, 0)")
ax.plot(0, 1, 0, "gH", markersize=15, label="Vertex (0, 1, 0)")
ax.plot(0, 0, 1, "bH", markersize=15, label="Vertex (0, 0, 1)")
ax.plot([0.8, 0.1, 0.1, 0.8], [0.1, 0.8, 0.1, 0.1], "k*-", label='Triangle')
ax.annotate("Convergence", (1.0/3.0, 1.0/3.0), xytext=(30,50),
            textcoords="offset points", arrowprops=dict(facecolor='black',
            width = 0.2, headwidth=8.0, shrink=0.05))
ax.plot(b, l, "m")
ax.text(0.9, 0.05, 'lower left')
ax.set_xlabel("Quantity b$\,/\,mol\,mol^{-1}$")
ax.set_ylabel("Quantity l$\,/\,mol\,mol^{-1}$")
ax.set_rlabel("Quantity r$\,/\,mol\,mol^{-1}$")
ax.legend()
ax.fill_between(l, l, y2=0, color='b')


# Data 2 (based on data posted by C. P. H. Lewis as triangleplot_demo.csv:
# http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg10051.html)
soil = {'sand': np.array([ 0.82,  0.17,  0.8 ,  0.63,  0.5 ,  0.0 ,  0.3 ,
                           0.3 ,  0.73,  0.03,  0.77]),
        'silt': np.array([ 0.04,  0.11,  0.14,  0.3 ,  0.4 ,  0.81,  0.41,
                           0.33,  0.03,  0.37,  0.21]),
        'clay': np.array([ 0.14,  0.72,  0.06,  0.07,  0.1 ,  0.19,  0.29,
                           0.37,  0.24,  0.60,  0.02]),
        'organic matter': np.array([ 0.45,  0.59,  0.35,  0.41,  0.07,
                                     0.22,  0.31,  0.21,  0.4 ,  0.27,
                                     0.2 ]),
        'porosity': np.array([ 0.48,  0.32,  0.23,  0.22,  0.62,  0.35,
                               0.43,  0.27,  0.44,  0.23,  0.53]),
        'site': ('Ger', 'Ma', 'Ss', 'Ss2', 'Din', 'Pro', 'Esn', 'Tra',
                 'Ffg', 'Rob', 'Aqw')}


# Example 2 -- Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='ternary')
s = ax.scatter(b=soil['sand'], l=soil['silt'], r=soil['clay'],
               c=soil['organic matter'], s=100.0*soil['porosity'], marker='o')
ax.set_title("Soil Plot\nSize of marker increases with porosity")
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("Sand / 1")
ax.set_ylabel("Silt / 1")
ax.set_rlabel("Clay / 1")
cax = ax.colorbar(mappable=s)
cax.set_label("Organic Matter / 1")


# Example 3 -- Patch
fig = plt.figure()
ax = fig.add_subplot(111, projection='ternary')

# Define the path for the patch.
vertices = [
    (0.1, 0.1),
    (0.1, 0.8),
    (0.8, 0.1),
    (0.1, 0.1),
    (0., 0.), # ignored
    ]
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
path = Path(vertices, codes)
ax.add_patch(PathPatch(path, facecolor='r', lw=2, alpha=0.5,
             label='original'))
ax.set_title('A Patch')


plt.show()
