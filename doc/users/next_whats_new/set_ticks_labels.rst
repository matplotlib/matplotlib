Settings tick positions and labels simultaneously in ``set_ticks``
------------------------------------------------------------------
`.Axis.set_ticks` (and the corresponding `.Axes.set_xticks` /
`.Axes.set_yticks`) got a new parameter *labels* allowing to set tick positions
and labels simultaneously.

Previously, setting tick labels was done using `.Axis.set_ticklabels` (or
the corresponding `.Axes.set_xticklabels` / `.Axes.set_yticklabels`). This
usually only makes sense if you previously fix the position with
`~.Axis.set_ticks`. Both functionality is now available in `~.Axis.set_ticks`.
The use of `.Axis.set_ticklabels` is discouraged, but it will stay available
for backward compatibility.

Note: This addition makes the API of `~.Axis.set_ticks` also more similar to
`.pyplot.xticks` / `.pyplot.yticks`, which already had the additional *labels*
parameter.
