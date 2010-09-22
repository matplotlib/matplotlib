"""
Show examples of matplotlib artists
http://matplotlib.sourceforge.net/api/artist_api.html

Several examples of standard matplotlib graphics primitives (artists)
are drawn using matplotlib API. Full list of artists and the
documentation is available at
http://matplotlib.sourceforge.net/api/artist_api.html

Copyright (c) 2010, Bartosz Telenczuk

License: This work is licensed under the BSD. A copy should be
included with this source code, and is also available at
http://www.opensource.org/licenses/bsd-license.php 
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

font = "sans-serif"
fig = plt.figure(figsize=(5,5))
ax = plt.axes([0,0,1,1])

# create 3x3 grid to plot the artists
pos = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1)

patches = []

# add a circle
art = mpatches.Circle(pos[:,0], 0.1,ec="none")
patches.append(art)
plt.text(pos[0,0], pos[1,0]-0.15, "Circle", ha="center",
        family=font, size=14)

# add a rectangle
art = mpatches.Rectangle(pos[:,1] - np.array([0.025, 0.05]), 0.05, 0.1,
        ec="none")
patches.append(art)
plt.text(pos[0,1], pos[1,1]-0.15, "Rectangle", ha="center",
        family=font, size=14)

# add a wedge
wedge = mpatches.Wedge(pos[:,2], 0.1, 30, 270, ec="none")
patches.append(wedge)
plt.text(pos[0,2], pos[1,2]-0.15, "Wedge", ha="center",
        family=font, size=14)

# add a Polygon
polygon = mpatches.RegularPolygon(pos[:,3], 5, 0.1)
patches.append(polygon)
plt.text(pos[0,3], pos[1,3]-0.15, "Polygon", ha="center",
        family=font, size=14)

#add an ellipse
ellipse = mpatches.Ellipse(pos[:,4], 0.2, 0.1)
patches.append(ellipse)
plt.text(pos[0,4], pos[1,4]-0.15, "Ellipse", ha="center",
        family=font, size=14)

#add an arrow
arrow = mpatches.Arrow(pos[0,5]-0.05, pos[1,5]-0.05, 0.1, 0.1, width=0.1)
patches.append(arrow)
plt.text(pos[0,5], pos[1,5]-0.15, "Arrow", ha="center",
        family=font, size=14)

# add a path patch 
Path = mpath.Path
verts = np.array([
     (0.158, -0.257),
     (0.035, -0.11),
     (-0.175, 0.20),
     (0.0375, 0.20),
     (0.085, 0.115),
     (0.22, 0.32),
     (0.3, 0.005),
     (0.20, -0.05),
     (0.158, -0.257),
    ])
verts = verts-verts.mean(0)
codes = [Path.MOVETO, 
         Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
         Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]

path = mpath.Path(verts/2.5+pos[:,6], codes)
patch = mpatches.PathPatch(path)
patches.append(patch)
plt.text(pos[0,6], pos[1,6]-0.15, "PathPatch", ha="center",
        family=font, size=14)

# add a fancy box
fancybox = mpatches.FancyBboxPatch(
        pos[:,7]-np.array([0.025, 0.05]), 0.05, 0.1, 
        boxstyle=mpatches.BoxStyle("Round", pad=0.02))
patches.append(fancybox)
plt.text(pos[0,7], pos[1,7]-0.15, "FancyBoxPatch", ha="center",
        family=font, size=14)

# add a line
x,y = np.array([[-0.06, 0.0, 0.1], [0.05,-0.05, 0.05]])
line = mlines.Line2D(x+pos[0,8], y+pos[1,8], lw=5.,
        alpha=0.4)
plt.text(pos[0,8], pos[1,8]-0.15, "Line2D", ha="center",
        family=font, size=14)

colors = 100*np.random.rand(len(patches))
collection = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
collection.set_array(np.array(colors))
ax.add_collection(collection)
ax.add_line(line)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
