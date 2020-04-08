"""
=================================
Inserting quiverkey into a legend
=================================

Demonstrates options for inserting a `~.axes.Axes.quiverkey` into a legend.

For a simple quiverkey demo, refer to
:doc:`/gallery/images_contours_and_fields/quiver_simple_demo`.

Please ensure you are familiar with the legend documentation
before proceeding with this demo: :func:`~matplotlib.pyplot.legend`.
"""
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

###############################################################################
# Adding the quiver key to the legend
# ===================================
#
# To add the quiverkey to a legend, simply pass it as a handle
# to :func:`~matplotlib.pyplot.legend`.
#
# Note that the quiverkey coordinates provided via the X and Y
# parameters args to `~.axes.Axes.quiverkey` are ignored.

fig, ax1 = plt.subplots()
q = ax1.quiver(X, Y, U, V)
qk1 = ax1.quiverkey(q, X=0.7, Y=0.1, U=10,
                    label='Quiver key: length = 10')


ax1.legend(handles=[qk1], loc='upper center')
ax1.set_title('Quiver key is placed in the legend')
plt.show()

###############################################################################
# Customizing the quiver key
# ==========================
#
# The other (non-positional) git parameters that `~.axes.Axes.quiverkey`
# accepts are preserved, however.

fig, ax2 = plt.subplots()
M = np.hypot(U, V)
q = ax2.quiver(X, Y, U, V, M)

qk2 = ax2.quiverkey(q, X=0, Y=0, U=10, angle=45.0,
                    label=r'Quiver key: length = 10, 45$^\circ$ angle')
qk3 = ax2.quiverkey(q, X=0, Y=0, U=10, color='purple', linewidth=2,
                    label='Quiver key: color = purple; linewidth=2')

ax2.legend(handles=[qk2, qk3], loc='upper center')
ax2.set_title('Quiver keys in the legend can be customized')
plt.show()

###############################################################################
# Adding large quiver keys to the legend
# ======================================
#
# When adding long quiver keys to a legend, you may need to tweak the
# handlelength, borderpad or other parameters of
# `~matplotlib.pyplot.legend` to make the arrow fit in the legend.

fig3, ax3 = plt.subplots()
q = ax3.quiver(X, Y, U, V, M)
qk = ax3.quiverkey(q, X=0, Y=0, U=25, angle=45.0,
                   label=r'Quiver key: length = 25, 45$^\circ$ angle')

ax3.legend(handles=[qk], loc='upper center', handlelength=2, borderpad=1.2)

ax3.set_title('Use legend\'s parameters to fit larger quiver keys')

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions and methods is shown in this example:

import matplotlib
matplotlib.axes.Axes.quiver
matplotlib.pyplot.quiver
matplotlib.axes.Axes.quiverkey
matplotlib.pyplot.quiverkey
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
