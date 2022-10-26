"""
===================================
Angle annotations on bracket arrows
===================================

This example shows how to add angle annotations to bracket arrow styles
created using `.FancyArrowPatch`. *angleA* and *angleB* are measured from a
vertical line as positive (to the left) or negative (to the right). Blue
`.FancyArrowPatch` arrows indicate the directions of *angleA* and *angleB*
from the vertical and axes text annotate the angle sizes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def get_point_of_rotated_vertical(origin, line_length, degrees):
    """Return xy coordinates of the vertical line end rotated by degrees."""
    rad = np.deg2rad(-degrees)
    return [origin[0] + line_length * np.sin(rad),
            origin[1] + line_length * np.cos(rad)]


fig, ax = plt.subplots(figsize=(8, 7))
ax.set(xlim=(0, 6), ylim=(-1, 4))
ax.set_title("Orientation of the bracket arrows relative to angleA and angleB")

for i, style in enumerate(["]-[", "|-|"]):
    for j, angle in enumerate([-40, 60]):
        y = 2*i + j
        arrow_centers = ((1, y), (5, y))
        vlines = ((1, y + 0.5), (5, y + 0.5))
        anglesAB = (angle, -angle)
        bracketstyle = f"{style}, angleA={anglesAB[0]}, angleB={anglesAB[1]}"
        bracket = FancyArrowPatch(*arrow_centers, arrowstyle=bracketstyle,
                                  mutation_scale=42)
        ax.add_patch(bracket)
        ax.text(3, y + 0.05, bracketstyle, ha="center", va="bottom")
        ax.vlines([i[0] for i in vlines], [y, y], [i[1] for i in vlines],
                  linestyles="--", color="C0")
        # Get the top coordinates for the drawn patches at A and B
        patch_tops = [get_point_of_rotated_vertical(center, 0.5, angle)
                      for center, angle in zip(arrow_centers, anglesAB)]
        # Define the connection directions for the annotation arrows
        connection_dirs = (1, -1) if angle > 0 else (-1, 1)
        # Add arrows and annotation text
        arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8"
        for vline, dir, patch_top, angle in zip(vlines, connection_dirs,
                                                patch_tops, anglesAB):
            kw = dict(connectionstyle=f"arc3,rad={dir * 0.5}",
                      arrowstyle=arrowstyle, color="C0")
            ax.add_patch(FancyArrowPatch(vline, patch_top, **kw))
            ax.text(vline[0] - dir * 0.15, y + 0.3, f'{angle}°', ha="center",
                    va="center")

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.ArrowStyle`
