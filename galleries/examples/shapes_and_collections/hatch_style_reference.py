"""
=====================
Hatch style reference
=====================

Hatches can be added to most polygons in Matplotlib, including `~.Axes.bar`,
`~.Axes.fill_between`, `~.Axes.contourf`, and children of `~.patches.Polygon`.
They are currently supported in the PS, PDF, SVG, macosx, and Agg backends. The WX
and Cairo backends do not currently support hatching.

See also :doc:`/gallery/images_contours_and_fields/contourf_hatching` for
an example using `~.Axes.contourf`, and
:doc:`/gallery/shapes_and_collections/hatch_demo` for more usage examples.

"""
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))

hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


def hatches_plot(ax, h):
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch=h))
    ax.text(1, -0.5, f"' {h} '", size=15, ha="center")
    ax.axis('equal')
    ax.axis('off')

for ax, h in zip(axs.flat, hatches):
    hatches_plot(ax, h)

# %%
# Hatching patterns can be repeated to increase the density.

fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))

hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

for ax, h in zip(axs.flat, hatches):
    hatches_plot(ax, h)

# %%
# Hatching patterns can be combined to create additional patterns.

fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))

hatches = ['/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']

for ax, h in zip(axs.flat, hatches):
    hatches_plot(ax, h)

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Rectangle`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.axes.Axes.text`
