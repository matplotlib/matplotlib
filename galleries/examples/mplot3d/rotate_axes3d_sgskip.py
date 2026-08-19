"""
==================
Rotating a 3D plot
==================

A very simple animation of a rotating 3D plot about all three axes.

See :doc:`wire3d_animation` for another example of animating a 3D plot.

(This example is skipped when building the documentation gallery because it
intentionally takes a long time to run)
"""

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
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180

    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    elev = azim = roll = 0
    if angle <= 360:
        elev = angle_norm
    elif angle <= 360*2:
        azim = angle_norm
    elif angle <= 360*3:
        roll = angle_norm
    else:
        elev = azim = roll = angle_norm

    # Update the axis view and title
    ax.view_init(elev, azim, roll)
    ax.set_title(f'Elevation: {elev}°, Azimuth: {azim}°, Roll: {roll}°')


ani = animation.FuncAnimation(fig, animate, interval=25, frames=360*4)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    component: animation,
#    level: advanced,
#    internal: high-bandwidth
