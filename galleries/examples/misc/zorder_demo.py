"""
===========
Zorder Demo
===========

The drawing order of artists is determined by their ``zorder`` attribute, which
is a floating point number. Artists with higher ``zorder`` are drawn on top.
You can change the order for individual artists by setting their ``zorder``.
The default value depends on the type of the Artist:

================================================================    =======
Artist                                                              Z-order
================================================================    =======
Images (`.AxesImage`, `.FigureImage`, `.BboxImage`)                 0
`.Patch`, `.PatchCollection`                                        1
`.Line2D`, `.LineCollection` (including minor ticks, grid lines)    2
Major ticks                                                         2.01
`.Text` (including Axes labels and titles)                          3
`.Legend`                                                           5
================================================================    =======

Any call to a plotting method can set a value for the zorder of that particular
item explicitly.

.. note::

   `~.axes.Axes.set_axisbelow` and :rc:`axes.axisbelow` are convenient helpers
   for setting the zorder of ticks and grid lines.

Drawing is done per `~.axes.Axes` at a time. If you have overlapping Axes, all
elements of the second Axes are drawn on top of the first Axes, irrespective of
their relative zorder.
"""

import matplotlib.pyplot as plt
import numpy as np

r = np.linspace(0.3, 1, 30)
theta = np.linspace(0, 4*np.pi, 30)
x = r * np.sin(theta)
y = r * np.cos(theta)

# %%
# The following example contains a `.Line2D` created by `~.axes.Axes.plot()`
# and the dots (a `.PatchCollection`) created by `~.axes.Axes.scatter()`.
# Hence, by default the dots are below the line (first subplot).
# In the second subplot, the ``zorder`` is set explicitly to move the dots
# on top of the line.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))

ax1.plot(x, y, 'C3', lw=3)
ax1.scatter(x, y, s=120)
ax1.set_title('Lines on top of dots')

ax2.plot(x, y, 'C3', lw=3)
ax2.scatter(x, y, s=120, zorder=2.5)  # move dots on top of line
ax2.set_title('Dots on top of lines')

plt.tight_layout()

# %%
# Many functions that create a visible object accepts a ``zorder`` parameter.
# Alternatively, you can call ``set_zorder()`` on the created object later.

x = np.linspace(0, 7.5, 100)
plt.rcParams['lines.linewidth'] = 5
plt.figure()
plt.plot(x, np.sin(x), label='zorder=2', zorder=2)  # bottom
plt.plot(x, np.sin(x+0.5), label='zorder=3',  zorder=3)
plt.axhline(0, label='zorder=2.5', color='lightgrey', zorder=2.5)
plt.title('Custom order of elements')
l = plt.legend(loc='upper right')
l.set_zorder(2.5)  # legend between blue and orange line
plt.show()
