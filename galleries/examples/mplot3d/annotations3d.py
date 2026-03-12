"""
========================
Annotating points in 3D
========================

The :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.annotate` method supports two
primary modes of annotation in 3D plots:

3D Data Annotations
   By passing a 3-tuple ``(x, y, z)`` to *xy* (and ensuring ``xycoords='data'``,
   which is the default), the annotation is anchored to a specific point in 3D
   space. As the 3D view is rotated or zoomed, the annotation is dynamically
   re-projected so it stays attached to the data point.

2D Screen Annotations
   By passing a standard 2-tuple ``(x, y)`` to *xy*, the annotation is treated
   as a standard 2D label on the canvas. This is highly useful for static UI
   elements, watermarks, or fixed text (e.g., using
   ``xycoords='axes fraction'``) that should remain stationary on the screen
   regardless of how the 3D axes are manipulated.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Create a simple 3D curve.
t = np.linspace(0, 2 * np.pi, 200)
x = np.cos(t)
y = np.sin(t)
z = t / (2 * np.pi)

idx = 60
xyz = (x[idx], y[idx], z[idx])

fig, axs = plt.subplots(
    1,
    2,
    figsize=(9, 4),
    constrained_layout=True,
    subplot_kw={"projection": "3d"},
)

views = [(25, -60), (25, 30)]

for ax, (elev, azim) in zip(axs, views):
    ax.plot(x, y, z, lw=2)
    ax.scatter(*xyz, s=40, color="C3")

    # Anchor in 3D data coordinates using a 3-tuple (x, y, z).  Use a 2D text
    # offset in points to keep the label readable.
    ax.annotate(
        "selected point",
        xyz,
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"elev={elev}, azim={azim}")

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner
