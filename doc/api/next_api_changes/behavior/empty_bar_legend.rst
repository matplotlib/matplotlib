Legend entry of an empty bar plot reflects the requested style
--------------------------------------------------------------

Calling `~matplotlib.axes.Axes.bar` with empty data previously produced a
legend entry with the default style and the first color of the property cycle.
The legend entry now reflects the requested bar style (e.g. *color*, *alpha*),
consistent with `~matplotlib.axes.Axes.plot` and `~matplotlib.axes.Axes.scatter`.
