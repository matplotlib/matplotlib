Lines now accept ``MarkerStyle`` instances as input
---------------------------------------------------
Similar to `~.Axes.scatter`, `~.Axes.plot` and `~.lines.Line2D` now accept
`~.markers.MarkerStyle` instances as input for the *marker* parameter::

    plt.plot(..., marker=matplotlib.markers.MarkerStyle("D"))

