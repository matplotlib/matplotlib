Standard getters/setters for axis inversion state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Whether an axis is inverted can now be queried and set using the `.axes.Axes`
getters `~.Axes.get_xinverted`/`~.Axes.get_yinverted` and setters
`~.Axes.set_xinverted`/`~.Axes.set_yinverted`.

The previously existing methods (`.Axes.xaxis_inverted`, `.Axes.invert_xaxis`)
are now discouraged (but not deprecated) due to their non-standard naming and
behavior.
