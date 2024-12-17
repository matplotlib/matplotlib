"""
==========
Streamplot
==========

A stream plot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the `~.axes.Axes.streamplot` function:

* Varying the color along a streamline.
* Varying the density of streamlines.
* Varying the line width along a streamline.
* Controlling the starting points of streamlines.
* Streamlines skipping masked regions and NaN values.
* Unbroken streamlines even when exceeding the limit of lines within a single
  grid cell.
"""
import matplotlib.pyplot as plt
import numpy as np

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U**2 + V**2)

fig, axs = plt.subplots(4, 2, figsize=(7, 12), height_ratios=[1, 1, 1, 2])
axs = axs.flat

#  Varying density along a streamline
axs[0].streamplot(X, Y, U, V, density=[0.5, 1])
axs[0].set_title('Varying Density')

# Varying color along a streamline
strm = axs[1].streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
fig.colorbar(strm.lines)
axs[1].set_title('Varying Color')

#  Varying line width along a streamline
lw = 5*speed / speed.max()
axs[2].streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw, num_arrows=5)
axs[2].set_title('Varying Line Width')

# Controlling the starting points of the streamlines
seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

strm = axs[3].streamplot(X, Y, U, V, color=U, linewidth=2,
                         cmap='autumn', start_points=seed_points.T)
fig.colorbar(strm.lines)
axs[3].set_title('Controlling Starting Points')

# Displaying the starting points with blue symbols.
axs[3].plot(seed_points[0], seed_points[1], 'bo')
axs[3].set(xlim=(-w, w), ylim=(-w, w))

# Adding more than one arrow to each streamline
axs[4].streamplot(X, Y, U, V, num_arrows=3)
axs[4].set_title('Multiple arrows')

axs[5].axis("off")

# Create a mask
mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = True
U[:20, :20] = np.nan
U = np.ma.array(U, mask=mask)

axs[6].streamplot(X, Y, U, V, color='r')
axs[6].set_title('Streamplot with Masking')

axs[6].imshow(~mask, extent=(-w, w, -w, w), alpha=0.5, cmap='gray',
              aspect='auto')
axs[6].set_aspect('equal')

axs[7].streamplot(X, Y, U, V, broken_streamlines=False)
axs[7].set_title('Streamplot with unbroken streamlines')

plt.tight_layout()
plt.show()

# %%
# Streamline computation
# ----------------------
#
# The streamlines are computed by integrating along the provided vector field
# from the seed points, which are either generated automatically or manually
# specified. The accuracy and smoothness of the streamlines can be adjusted using
# the ``integration_max_step`` and ``integration_max_error`` optional parameters.
# See the `~.axes.Axes.streamplot` function documentation for more details.
#
# This example shows how adjusting the maximum allowed step size and error for
# the integrator changes the appearance of the streamline. The differences can
# be subtle, but can be observed particularly where the streamlines have
# high curvature.

# Linear potential flow over a lifting cylinder
n = 50
x, y = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-3, 3, n))
th = np.arctan2(y, x)
r = np.sqrt(x**2 + y**2)
vr = -np.cos(th) / r**2
vt = -np.sin(th) / r**2 - 1 / r
vx = vr * np.cos(th) - vt * np.sin(th) + 1.0
vy = vr * np.sin(th) + vt * np.cos(th)

# Seed points
n_seed = 50
seed_pts = np.column_stack((np.full(n_seed, -1.75), np.linspace(-2, 2, n_seed)))

_, axs = plt.subplots(2, 1, figsize=(6, 9.5))
for i, max_val in enumerate([5, 0.05]):
    axs[i].streamplot(
        x,
        y,
        vx,
        vy,
        start_points=seed_pts,
        broken_streamlines=False,
        density=1,
        arrowsize=1e-10,
        linewidth=0.5,
        color="k",
        integration_max_step=max_val,
        integration_max_error=max_val,
    )
    axs[i].text(
        0.0,
        0.0,
        f"integration_max_step: {max_val}\n" +
        f"integration_max_error: {max_val}",
        ha="center",
        va="center",
    )

# Draw the cylinder
th_circ = np.linspace(0, 2 * np.pi, 100)
for ax in axs:
    ax.fill(np.cos(th_circ), np.sin(th_circ), color="w", ec="k")

    ax.set_aspect("equal")
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

plt.tight_layout()
plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.streamplot` / `matplotlib.pyplot.streamplot`
#    - `matplotlib.gridspec.GridSpec`
