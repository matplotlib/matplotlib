"""
===================================
Position of the arrowhead in quiver
===================================

The *head_pos* parameter of `~.axes.Axes.quiver` sets where the arrowhead sits
along the shaft. It can be 'tail', 'middle' or 'tip' (the default), or a number
between 0 (the head starts at the tail) and 1 (the head ends at the tip).

Each row below draws the same arrows with a different *head_pos*. The
arrowhead position is independent of *pivot*, which sets the part of the arrow
that is anchored at the *X*, *Y* position, and the quiver key uses the same
arrowhead position as its quiver.

For more advanced options refer to
:doc:`/gallery/images_contours_and_fields/quiver_demo`.
"""
import matplotlib.pyplot as plt
import numpy as np

head_positions = ['tail', 0.25, 'middle', 0.75, 'tip']

angles = np.linspace(0, 270, 6)
lengths = np.linspace(1, 0.5, 6)
x = np.arange(len(angles))
u = lengths * np.cos(np.deg2rad(angles))
v = lengths * np.sin(np.deg2rad(angles))

fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
rows = np.arange(len(head_positions))[::-1]  # First row at the top.
for y, head_pos in zip(rows, head_positions):
    q = ax.quiver(x, np.full_like(x, y), u, v, head_pos=head_pos,
                  pivot='middle', angles='xy', scale_units='xy', scale=1.25,
                  width=0.008)
ax.quiverkey(q, X=0.8, Y=1.03, U=1, label='length 1', labelpos='E')
ax.set(xlim=(-0.7, len(x) - 0.3), ylim=(-0.6, len(rows) - 0.4),
       xticks=[], yticks=rows,
       yticklabels=[f'head_pos={p!r}' for p in head_positions],
       aspect='equal')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.quiver` / `matplotlib.pyplot.quiver`
#    - `matplotlib.axes.Axes.quiverkey` / `matplotlib.pyplot.quiverkey`
#
# .. tags::
#
#    component: axes
#    component: quiver
#    styling: position
#    level: beginner
