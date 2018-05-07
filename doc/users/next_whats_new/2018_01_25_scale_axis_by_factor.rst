Ability to scale axis by a fixed order of magnitude
---------------------------------------------------

To scale an axis by a fixed order of magnitude, set the *scilimits* argument of
``Axes.ticklabel_format`` to the same (non-zero) lower and upper limits. Say to scale
the y axis by a million (1e6), use ``ax.ticklabel_format(style='sci', scilimits=(6, 6), axis='y')``.

The behavior of ``scilimits=(0, 0)`` is unchanged. With this setting, matplotlib will adjust
the order of magnitude depending on the axis values, rather than keeping it fixed. Previously, setting
``scilimits=(m, m)`` was equivalent to setting ``scilimits=(0, 0)``.
