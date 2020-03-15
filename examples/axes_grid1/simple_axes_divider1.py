"""
=====================
Simple Axes Divider 1
=====================

"""

from mpl_toolkits.axes_grid1 import Size, Divider
import matplotlib.pyplot as plt


##############################################################################
# Fixed axes sizes; fixed paddings.

fig = plt.figure(figsize=(6, 6))

# Sizes are in inches.
horiz = [Size.Fixed(1.), Size.Fixed(.5), Size.Fixed(1.5), Size.Fixed(.5)]
vert = [Size.Fixed(1.5), Size.Fixed(.5), Size.Fixed(1.)]

rect = (0.1, 0.1, 0.8, 0.8)
# Divide the axes rectangle into a grid with sizes specified by horiz * vert.
divider = Divider(fig, rect, horiz, vert, aspect=False)

# The rect parameter will actually be ignored and overridden by axes_locator.
ax1 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=0))
ax2 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=2))
ax3 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=2, ny=2))
ax4 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=2, nx1=4, ny=0))

for ax in fig.axes:
    ax.tick_params(labelbottom=False, labelleft=False)

##############################################################################
# Axes sizes that scale with the figure size; fixed paddings.

fig = plt.figure(figsize=(6, 6))

horiz = [Size.Scaled(1.5), Size.Fixed(.5), Size.Scaled(1.), Size.Scaled(.5)]
vert = [Size.Scaled(1.), Size.Fixed(.5), Size.Scaled(1.5)]

rect = (0.1, 0.1, 0.8, 0.8)
# Divide the axes rectangle into a grid with sizes specified by horiz * vert.
divider = Divider(fig, rect, horiz, vert, aspect=False)

# The rect parameter will actually be ignored and overridden by axes_locator.
ax1 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=0))
ax2 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=2))
ax3 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=2, ny=2))
ax4 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=2, nx1=4, ny=0))

for ax in fig.axes:
    ax.tick_params(labelbottom=False, labelleft=False)

plt.show()
