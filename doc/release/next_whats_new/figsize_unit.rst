Figure size units
-----------------

When creating figures, it is now possible to define figure sizes in cm or pixel.

Up to now the figure size is specified via ``plt.figure(..., figsize=(6, 4))``,
and the given numbers are interpreted as inches. It is now possible to add a
unit string to the tuple, i.e. ``plt.figure(..., figsize=(600, 400, "px"))``.
Supported unit strings are "in", "cm", "px".
