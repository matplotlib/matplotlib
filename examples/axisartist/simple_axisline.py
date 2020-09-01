"""
===============
Simple Axisline
===============

"""

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero


fig = plt.figure()
fig.subplots_adjust(right=0.85)
ax = SubplotZero(fig, 1, 1, 1)
fig.add_subplot(ax)

# make right and top axis invisible
ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)

# make xzero axis (horizontal axis line through y=0) visible.
ax.axis["xzero"].set_visible(True)
ax.axis["xzero"].label.set_text("Axis Zero")

ax.set_ylim(-2, 4)
ax.set_xlabel("Label X")
ax.set_ylabel("Label Y")
# or
#ax.axis["bottom"].label.set_text("Label X")
#ax.axis["left"].label.set_text("Label Y")

# make new (right-side) yaxis, but with some offset
ax.axis["right2"] = ax.new_fixed_axis(loc="right", offset=(20, 0))
ax.axis["right2"].label.set_text("Label Y2")

ax.plot([-2, 3, 2])
plt.show()
