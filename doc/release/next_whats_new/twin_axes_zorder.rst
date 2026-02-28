Twin Axes ``delta_zorder``
--------------------------

`~matplotlib.axes.Axes.twinx` and `~matplotlib.axes.Axes.twiny` now accept a
*delta_zorder* keyword argument, a relative offset added to the original Axes'
zorder, to control whether the twin Axes is drawn in front of, or behind, the
original Axes.  For example, pass ``delta_zorder=-1`` to easily draw a twin Axes
behind the main Axes.

In addition, Matplotlib now automatically manages background patch visibility
for each group of twinned Axes so that only the bottom-most Axes in the group
has a visible background patch (respecting ``frameon``).
