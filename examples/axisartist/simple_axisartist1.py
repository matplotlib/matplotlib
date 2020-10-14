"""
==================
Simple Axisartist1
==================

"""
import matplotlib.pyplot as plt
from mpl_toolkits import axisartist

fig = plt.figure()
fig.subplots_adjust(right=0.85)
ax = fig.add_subplot(axes_class=axisartist.Axes)

# make some axis invisible
ax.axis["bottom", "top", "right"].set_visible(False)

# make an new axis along the first axis axis (x-axis) which pass
# through y=0.
ax.axis["y=0"] = ax.new_floating_axis(nth_coord=0, value=0,
                                      axis_direction="bottom")
ax.axis["y=0"].toggle(all=True)
ax.axis["y=0"].label.set_text("y = 0")

ax.set_ylim(-2, 4)

plt.show()
