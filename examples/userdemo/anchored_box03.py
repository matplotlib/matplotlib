"""
==============
Anchored Box03
==============

"""
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox


fig, ax = plt.subplots(figsize=(3, 3))

box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
el = Ellipse((0, 0), width=0.1, height=0.4, angle=30)  # in data coordinates!
box.drawing_area.add_artist(el)

ax.add_artist(box)

plt.show()
