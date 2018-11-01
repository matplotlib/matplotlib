"""
=====================
Marker filling-styles
=====================

Reference for marker fill-styles included with Matplotlib.

Also refer to the
:doc:`/gallery/lines_bars_and_markers/marker_fillstyle_reference`
and :doc:`/gallery/shapes_and_collections/marker_path` examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


points = np.ones(5)  # Draw 5 points for each line
marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='tab:red')

fig, ax = plt.subplots()

# Plot all fill styles.
for y, fill_style in enumerate(Line2D.fillStyles):
    ax.text(-0.5, y, repr(fill_style),
            horizontalalignment='center', verticalalignment='center')
    ax.plot(y * points, fillstyle=fill_style, **marker_style)

ax.set_axis_off()
ax.set_title('fill style')

plt.show()
