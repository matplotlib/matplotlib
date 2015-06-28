Mostly unified linestyles for `Line2D`, `Patche` and `Collection`
`````````````````````````````````````````````````````````````````

The handling of linestyles for Lines, Patches and Collections has been
unified.  Now they all support defining linestyles with short symbols,
like `"--"`, as well as with full names, like ``"dashed"``. Also the
definition using a dash pattern (``(0., [3., 3.])``) is supported for all
methods using `Line2D`, `Patche` or ``Collection*``.
