"""
========
Colorbar
========

Use `~.Figure.colorbar` by specifying the mappable object (here
the `.AxesImage` returned by `~.axes.Axes.imshow`)
and the Axes to attach the colorbar to.
"""

import matplotlib.pyplot as plt
import numpy as np

# setup some generic data
N = 37
x, y = np.mgrid[:N, :N]
Z = (np.cos(x*0.2) + np.sin(y*0.3))

# mask out the negative and positive values, respectively
Zpos = np.ma.masked_less(Z, 0)
Zneg = np.ma.masked_greater(Z, 0)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

# plot just the positive data and add a colorbar
ax1.imshow(Zpos, cmap='Blues', interpolation='none')
ax1.colorbar()

# repeat everything above for the negative data
# you can specify location, anchor and shrink the colorbar
ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
ax2.colorbar(location='right', anchor=(0, 0.3), shrink=0.7)

# Plot both positive and negative values between +/- 1.2
ax3.imshow(Z, cmap='RdBu', vmin=-1.2, vmax=1.2, interpolation='none')
# Add minorticks on the colorbar to make it easy to read the
# values off the colorbar.
cbar = ax3.colorbar(extend='both')
cbar.minorticks_on()
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colorbar.Colorbar.minorticks_on`
#    - `matplotlib.colorbar.Colorbar.minorticks_off`
#
# .. tags::
#
#    component: colorbar
#    styling: color
#    plot-type: imshow
#    level: beginner
