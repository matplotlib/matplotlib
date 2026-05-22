"""
=====================================
Clip the data to the axes view limits
=====================================

Demonstrate clipping of 3D data to the axes view limits.

Without ``axlim_clip``, data may extend beyond the axes view limits. With
``axlim_clip=True`` and ``axlim_clip_mode="hide"``, surface patches or line
segments with vertices outside the view limits are hidden. With
``axlim_clip_mode="clip"``, they are geometrically clipped to the axes
view-limit box.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

fig, axs = plt.subplots(
    2, 3,
    subplot_kw={"projection": "3d"},
    figsize=(10, 7),
    layout="constrained",
)

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
R = np.hypot(X, Y)
Z = np.sin(R)

xlim = (-3.2, 3.2)
ylim = (-3.0, 2.6)
zlim = (-0.45, 0.85)

cases = [
    ("unclipped", dict(axlim_clip=False)),
    ('axlim_clip_mode="hide"', dict(axlim_clip=True, axlim_clip_mode="hide")),
    ('axlim_clip_mode="clip"', dict(axlim_clip=True, axlim_clip_mode="clip")),
]

for ax, (title, clip_kwargs) in zip(axs[0], cases):
    ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        **clip_kwargs,
    )
    ax.set_title(f"surface\n{title}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

for ax, (title, clip_kwargs) in zip(axs[1], cases):
    ax.plot_wireframe(
        X, Y, Z,
        rstride=2,
        cstride=2,
        linewidth=0.6,
        **clip_kwargs,
    )
    ax.set_title(f"wireframe\n{title}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

plt.show()
# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner
