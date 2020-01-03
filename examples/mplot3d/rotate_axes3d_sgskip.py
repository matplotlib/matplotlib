"""
==================
Rotating a 3D plot
==================

A very simple animation of a rotating 3D plot.

See wire3d_animation_demo for another simple example of animating a 3D plot.

(This example is skipped when building the documentation gallery because it
intentionally takes a long time to run)
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# load some test data for demonstration and plot a wireframe
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
