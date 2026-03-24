"""
========================
Annotating points in 3D
========================

Demonstrates how to annotate 3D plots using 2D and 3D coordinates.

Both the anchor point (``xy``) and the text position (``xytext``) can be
specified using 3D data coordinates (a 3-tuple) or 2D screen coordinates (a
2-tuple).

When using 3D coordinates, set the coordinate system to ``'data'``. The
annotation will dynamically re-project as the 3D view is rotated. Mixing 2D and
3D endpoints allows for labels that stay fixed on the screen while tracking a
point in 3D space.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Create a simple 3D curve.
t = np.linspace(0, 2 * np.pi, 200)
x = np.cos(t)
y = np.sin(t)
z = t / (2 * np.pi)

# Select points for annotation.
point_3d = (x[60], y[60], z[60])
text_pos_3d = (x[150], y[150], z[150] + 0.12)

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 5),
    subplot_kw={'projection': '3d'},
    constrained_layout=True
)

for ax in (ax1, ax2):
    ax.plot(x, y, z, lw=2, alpha=0.7)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")

# --- Subplot 1: 3D Data Anchor + 2D Screen Offset ---
# Common use case: label tracks a point but stays readable and fixed on screen.
ax1.scatter(*point_3d, color="C3", s=40)
ax1.annotate(
    "3D Anchor\n2D Offset",
    xy=point_3d,
    xytext=(35, 35),
    textcoords="offset points",
    ha="left",
    va="bottom",
    arrowprops=dict(arrowstyle="->", lw=1),
)
ax1.set_title("3D Data Anchor + 2D Offset\n(Text stays fixed on screen)")

# --- Subplot 2: 3D Data Anchor + 3D Text Position ---
# Demonstrates that the text position itself can live in the 3D data space.
ax2.scatter(*point_3d, color="C3", s=40)
ax2.scatter(*text_pos_3d, color="C1", s=40)
ax2.annotate(
    "3D Anchor\n3D Text Position",
    xy=point_3d,
    xytext=text_pos_3d,
    textcoords="data",
    ha="left",
    va="bottom",
    arrowprops=dict(arrowstyle="->", lw=1, color="C1"),
)
ax2.set_title("3D Data Anchor + 3D Text\n(Both orbit in 3D space)")

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner
