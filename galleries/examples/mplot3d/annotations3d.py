"""
========================
Annotating points in 3D
========================

`~mpl_toolkits.mplot3d.axes3d.Axes3D.annotate` supports anchoring annotations in
3D data coordinates by passing ``(x, y, z)`` for *xy* (with ``xycoords='data'``,
the default).  The annotation is projected during draws, so it stays attached
to the point as the view changes.
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
