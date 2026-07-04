"""
==================
Rotating a 3D plot
==================

A very simple animation of a rotating 3D plot about all three axes.

See :doc:`wire3d_animation` for another example of animating a 3D plot.

(This example is skipped when building the documentation gallery because it
intentionally takes a long time to run)
"""

# sphinx_gallery_thumbnail_path = "_static/rotate_axes3d.png"
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some example data and plot a basic wireframe.
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# Set the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# Rotate the axes and update
def animate(angle):
    # Normalize the angle to the range
    angle_norm = (angle + 180) % 360 - 180
    # Update the axis view
    ax.view_init(elev=20, azim=angle_norm)
    return fig,


# Create the animation
anim = animation.FuncAnimation(fig, animate, frames=360, interval=30, blit=True)

plt.show()
