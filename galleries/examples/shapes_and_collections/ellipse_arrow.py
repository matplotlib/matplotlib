"""
===================================
Ellipse with orientation arrow Demo
===================================

This demo shows how to draw an ellipse with an orientation arrow.
Compare this to the :doc:`Ellipse collection example
</gallery/shapes_and_collections/ellipse_collection>`.
"""

# Import of namespaces
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

import numpy as np

from typing import Tuple


def getMinorMajor(ellipse: Ellipse) -> Tuple[list, list]:
    """
    Calculates the end points of minor and major axis of an ellipse.

    Parameters
    ----------
    ellipse : ~matplotlib.patches.Ellipse
        Ellipse patch.

    Returns
    -------
    ~typing.Tuple[list, list]
    """
    # Calculate the endpoints of the minor axis
    x0_minor = ellipse.center[0] - ellipse.height / 2 * np.sin(
        np.deg2rad(ellipse.angle)
    )
    y0_minor = ellipse.center[1] + ellipse.height / 2 * np.cos(
        np.deg2rad(ellipse.angle)
    )
    x1_minor = ellipse.center[0] + ellipse.height / 2 * np.sin(
        np.deg2rad(ellipse.angle)
    )
    y1_minor = ellipse.center[1] - ellipse.height / 2 * np.cos(
        np.deg2rad(ellipse.angle)
    )

    # Calculate the endpoints of the major axis
    x0_major = ellipse.center[0] - ellipse.width / 2 * np.cos(np.deg2rad(ellipse.angle))
    y0_major = ellipse.center[1] - ellipse.width / 2 * np.sin(np.deg2rad(ellipse.angle))
    x1_major = ellipse.center[0] + ellipse.width / 2 * np.cos(np.deg2rad(ellipse.angle))
    y1_major = ellipse.center[1] + ellipse.width / 2 * np.sin(np.deg2rad(ellipse.angle))
    return [(x0_minor, y0_minor), (x1_minor, y1_minor)], [
        (x0_major, y0_major),
        (x1_major, y1_major),
    ]


# Define the ellipse
center = (2, 4)
width = 30
height = 20
angle = 35
ellipse = Ellipse(
    xy=center,
    width=width,
    height=height,
    angle=angle,
    facecolor="none",
    edgecolor="b",
)

minor, major = getMinorMajor(ellipse)

# Create a figure and axis
fig, ax = plt.subplots(1, 1, subplot_kw={"aspect": "equal"})

# Add the ellipse patch to the axis
ax.add_patch(ellipse)

# Plot a arrow marker at the end point of minor axis
t = Affine2D().rotate_deg(angle)
ax.plot(
    minor[0][0],
    minor[0][1],
    color="b",
    marker=MarkerStyle(">", "full", t),
    markersize=10,
)

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Ellipse`
