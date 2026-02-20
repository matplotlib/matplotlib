Twin Axes ``zorder``
--------------------

`~matplotlib.axes.Axes.twinx` and `~matplotlib.axes.Axes.twiny` now accept a
*zorder* keyword argument to control whether the twin Axes is drawn in front of,
or behind, the original Axes.

In addition, Matplotlib now automatically manages background patch visibility
for each group of twinned Axes so that only the bottom-most Axes in the group
has a visible background patch (respecting ``frameon``).
